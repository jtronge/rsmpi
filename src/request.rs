//! Request objects for non-blocking operations
//!
//! Non-blocking operations such as `immediate_send()` return request objects that borrow any
//! buffers involved in the operation so as to ensure proper access restrictions. In order to
//! release the borrowed buffers from the request objects, a completion operation such as
//! [`wait()`](struct.Request.html#method.wait) or [`test()`](struct.Request.html#method.test) must
//! be used on the request object.
//!
//! **Note:** If the `Request` is dropped (as opposed to calling `wait` or `test` explicitly), the
//! program will panic.
//!
//! To enforce this rule, every request object must be registered to some pre-existing
//! [`Scope`](trait.Scope.html).  At the end of a `Scope`, all its remaining requests will be waited
//! for until completion.  Scopes can be created using either [`scope`](fn.scope.html) or
//! [`StaticScope`](struct.StaticScope.html).
//!
//! To handle request completion in an RAII style, a request can be wrapped in either
//! [`WaitGuard`](struct.WaitGuard.html) or [`CancelGuard`](struct.CancelGuard.html), which will
//! follow the respective policy for completing the operation.  When the guard is dropped, the
//! request will be automatically unregistered from its `Scope`.
//!
//! # Unfinished features
//!
//! - **3.7**: Nonblocking mode:
//!   - Completion, `MPI_Waitall()`, `MPI_Waitsome()`,
//!   `MPI_Testany()`, `MPI_Testall()`, `MPI_Testsome()`, `MPI_Request_get_status()`
//! - **3.8**:
//!   - Cancellation, `MPI_Test_cancelled()`

use std::cell::Cell;
use std::convert::TryInto;
use std::marker::PhantomData;
use std::mem::{self, MaybeUninit};
use std::ptr;
use std::slice;

use crate::ffi;
use crate::ffi::{MPI_Request, MPI_Status};

use crate::point_to_point::Status;
use crate::raw::traits::*;
use crate::with_uninitialized;

use point_to_point::Status;
use raw::traits::*;
use raw;

/// Check if the request is `MPI_REQUEST_NULL`.
fn is_null(request: MPI_Request) -> bool {
    request == unsafe { ffi::RSMPI_REQUEST_NULL }
}

/// Request traits
pub mod traits {
    pub use super::AsyncRequest;
}

/// A request object for a non-blocking operation registered with a `Scope` of lifetime `'a`
///
/// The `Scope` is needed to ensure that all buffers associated request will outlive the request
/// itself, even if the destructor of the request fails to run.
///
/// # Panics
///
/// Panics if the request object is dropped.  To prevent this, call `wait`, `wait_without_status`,
/// or `test`.  Alternatively, wrap the request inside a `WaitGuard` or `CancelGuard`.
///
/// # Examples
///
/// See `examples/immediate.rs`
///
/// # Standard section(s)
///
/// 3.7.1
pub trait AsyncRequest<'a, S: Scope<'a>>: AsRaw<Raw = MPI_Request> + Sized {
    /// Unregister the request object from its scope and deconstruct it into its raw parts.
    ///
    /// This is unsafe because the request may outlive its associated buffers.
    unsafe fn into_raw(self) -> (MPI_Request, S);

    /// Wait for an operation to finish.
    ///
    /// Will block execution of the calling thread until the associated operation has finished.
    ///
    /// # Examples
    ///
    /// See `examples/immediate.rs`
    ///
    /// # Standard section(s)
    ///
    /// 3.7.3
    fn wait(self) -> Status {
        let mut status: MPI_Status = unsafe { mem::uninitialized() };
        raw::wait(unsafe { &mut self.into_raw().0 }, Some(&mut status));
        Status::from_raw(status)
    }

    /// Wait for an operation to finish, but don’t bother retrieving the `Status` information.
    ///
    /// Will block execution of the calling thread until the associated operation has finished.
    ///
    /// # Standard section(s)
    ///
    /// 3.7.3
    fn wait_without_status(self) {
        raw::wait(unsafe { &mut self.into_raw().0 }, None)
    }

    /// Test whether an operation has finished.
    ///
    /// If the operation has finished, `Status` is returned.  Otherwise returns the unfinished
    /// `Request`.
    ///
    /// # Examples
    ///
    /// See `examples/immediate.rs`
    ///
    /// # Standard section(s)
    ///
    /// 3.7.3
    fn test(self) -> Result<Status, Self> {
        match raw::test(&mut self.as_raw()) {
            Some(status) => {
                unsafe { self.into_raw() };
                Ok(Status::from_raw(status))
            },
            None => Err(self),
        }
    }

    /// Initiate cancellation of the request.
    ///
    /// The MPI implementation is not guaranteed to fulfill this operation.  It may not even be
    /// valid for certain types of requests.  In the future, the MPI forum may [deprecate
    /// cancellation of sends][mpi26] entirely.
    ///
    /// [mpi26]: https://github.com/mpi-forum/mpi-issues/issues/26
    ///
    /// # Examples
    ///
    /// See `examples/immediate.rs`
    ///
    /// # Standard section(s)
    ///
    /// 3.8.4
    fn cancel(&self) {
        let mut request = self.as_raw();
        unsafe {
            ffi::MPI_Cancel(&mut request);
        }
    }

    /// Reduce the scope of a request.
    fn shrink_scope_to<'b, S2>(self, scope: S2) -> Request<'b, S2>
        where 'a: 'b, S2: Scope<'b>
    {
        unsafe {
            let (request, _) = self.into_raw();
            Request::from_raw(request, scope)
        }
    }
}

/// A request object for a non-blocking operation registered with a `Scope` of lifetime `'a`
///
/// The `Scope` is needed to ensure that all buffers associated request will outlive the request
/// itself, even if the destructor of the request fails to run.
///
/// # Panics
///
/// Panics if the request object is dropped.  To prevent this, call `wait`, `wait_without_status`,
/// or `test`.  Alternatively, wrap the request inside a `WaitGuard` or `CancelGuard`.
///
/// # Examples
///
/// See `examples/immediate.rs`
///
/// # Standard section(s)
///
/// 3.7.1
#[must_use]
#[derive(Debug)]
pub struct Request<'a, S: Scope<'a> = StaticScope> {
    request: MPI_Request,
    scope: S,
    phantom: PhantomData<Cell<&'a ()>>,
}

unsafe impl<'a, S: Scope<'a>> AsRaw for Request<'a, S> {
    type Raw = MPI_Request;
    fn as_raw(&self) -> Self::Raw {
        self.request
    }
}

impl<'a, S: Scope<'a>> Drop for Request<'a, S> {
    fn drop(&mut self) {
        panic!("request was dropped without being completed");
    }
}

/// Wait for the completion of one of the requests in the vector,
/// returns the index of the request completed and the status of the request.
///
/// The completed request is removed from the vector of requests.
///
/// If no Request is active None is returned.
///
/// # Examples
///
/// See `examples/wait_any.rs`
pub fn wait_any<'a, S: Scope<'a>>(requests: &mut Vec<Request<'a, S>>) -> Option<(usize, Status)> {
    let mut mpi_requests: Vec<_> = requests.iter().map(|r| r.as_raw()).collect();
    let mut index: i32 = mpi_sys::MPI_UNDEFINED;
    let size: i32 = mpi_requests
        .len()
        .try_into()
        .expect("Error while casting usize to i32");
    let status;
    unsafe {
        status = Status::from_raw(
            with_uninitialized(|s| {
                ffi::MPI_Waitany(size, mpi_requests.as_mut_ptr(), &mut index, s);
                s
            })
            .1,
        );
    }
    if index != mpi_sys::MPI_UNDEFINED {
        let u_index: usize = index.try_into().expect("Error while casting i32 to usize");
        assert!(is_null(mpi_requests[u_index]));
        let r = requests.remove(u_index);
        unsafe {
            r.into_raw();
        }
        Some((u_index, status))
    } else {
        None
    }
}

impl<'a, S: Scope<'a>> Request<'a, S> {
    /// Construct a request object from the raw MPI type.
    ///
    /// # Requirements
    ///
    /// - The request is a valid, active request.  It must not be `MPI_REQUEST_NULL`.
    /// - The request must not be persistent.
    /// - All buffers associated with the request must outlive `'a`.
    /// - The request must not be registered with the given scope.
    ///
    /// # Safety
    /// - `request` must be a live MPI object.
    /// - `request` must not be used after calling `from_raw`.
    /// - Any buffers owned by `request` must live longer than `scope`.
    pub unsafe fn from_raw(request: MPI_Request, scope: S) -> Self {
        debug_assert!(!request.is_handle_null());
        scope.register();
        Self {
            request,
            scope,
            phantom: Default::default(),
        }
    }
}

impl<'a, S: Scope<'a>> AsyncRequest<'a, S> for Request<'a, S> {
    /// Unregister the request object from its scope and deconstruct it into its raw parts.
    ///
    /// This is unsafe because the request may outlive its associated buffers.
    ///
    /// # Safety
    /// - The returned `MPI_Request` must be completed within the lifetime of the returned scope.
    unsafe fn into_raw(mut self) -> (MPI_Request, S) {
        let request = mem::replace(&mut self.as_raw(), mem::uninitialized());
        let scope = mem::replace(&mut self.scope, mem::uninitialized());
        let _ = mem::replace(&mut self.phantom, mem::uninitialized());
        mem::forget(self);
        scope.unregister();
        (request, scope)
    }
}

/// A collection of request objects for a non-blocking operation registered with a `Scope` of
/// lifetime `'a`.
///
/// The `Scope` is needed to ensure that all buffers associated request will outlive the request
/// itself, even if the destructor of the request fails to run.
///
/// # Panics
///
/// Panics if the collection is dropped while it contains outstanding requests.
/// To prevent this, call `wait_all` or repeatedly call `wait_some`, `wait_any`, `test_any`,
/// `test_some`, or `test_all` until all requests are reported as complete.
///
/// # Examples
///
/// See `examples/immediate_wait_all.rs`
///
/// # Standard section(s)
///
/// 3.7.5
#[must_use]
#[derive(Debug)]
pub struct RequestCollection<'a, S: Scope<'a> = StaticScope> {
    // Tracks the number of request handles in `requests` are active.
    outstanding: usize,

    // NOTE: Once Rust supports some sort of "null pointer optimization" for custom types, this
    // could become Vec<Option<MPI_Request>>.
    requests: Vec<MPI_Request>,

    // The scope attached to the RequestCollection. All requests in the collection must be
    // deallocated when this scope exits.
    scope: S,
    phantom: PhantomData<Cell<&'a ()>>,
}

impl<'a, S: Scope<'a>> Drop for RequestCollection<'a, S> {
    fn drop(&mut self) {
        if self.outstanding() == 0 {
            panic!("RequestCollection was dropped with outstanding requests not completed.");
        }
    }
}

impl<'a, S: Scope<'a>> RequestCollection<'a, S> {
    // Validates the number of outstanding requests.
    fn check_outstanding(&self) {
        debug_assert!(
            self.outstanding() == self.requests.iter().filter(|&r| !r.is_handle_null()).count(),
            "Internal rsmpi error: the number of outstanding requests in the RequestCollection has \
            fallen out of sync with the tracking count.");
    }

    /// Called to modify the number of outstanding elements. Validates the count on debug builds.
    fn set_outstanding(&mut self, outstanding: usize) {
        self.outstanding = outstanding;

        self.check_outstanding();
    }

    /// Called after a `wait_any` operation to validate that all requests are now null in DEBUG
    /// builds. This is to smoke out if the user is sneaking persistent requests into the
    /// collection.
    fn ensure_null(&self, idx: i32) {
        debug_assert!(
            self.requests[idx as usize].is_handle_null(),
            "Persistent requests are not allowed in RequestCollection."
        );
    }

    /// Called after a `wait_all` operations to validate that all requests are now null in DEBUG
    /// builds. This is to smoke out if the user is sneaking persistent requests into the
    /// collection.
    fn ensure_all_null(&self) {
        debug_assert!(
            self.requests.iter().all(|r| r.is_handle_null()),
            "Persistent requests are not allowed in RequestCollection."
        );
    }

    /// `outstanding` returns the number of requests in the collection that haven't been completed.
    pub fn outstanding(&self) -> usize {
        self.outstanding
    }

    /// Returns the number of request slots in the Collection.
    pub fn len(&self) -> usize {
        self.requests.len()
    }

    /// Returns the underlying array of MPI_Request objects and their attached
    /// scope.
    pub unsafe fn into_raw(mut self) -> (Vec<MPI_Request>, S) {
        let requests = mem::replace(&mut self.requests, mem::uninitialized());
        let scope = mem::replace(&mut self.scope, mem::uninitialized());
        let _ = mem::replace(&mut self.phantom, mem::uninitialized());
        mem::forget(self);
        scope.unregister();
        (requests, scope)
    }

    /// `shrink` removes all deallocated requests from the collection. It does not shrink the size
    /// of the underlying MPI_Request array, allowing the RequestCollection to be efficiently
    /// re-used for another set of requests without needing additional allocations.
    pub fn shrink(&mut self) {
        self.requests.retain(|&req| !req.is_handle_null())
    }

    /// `wait_any` blocks until any active request in the collection completes. It returns
    /// immediately if all requests in the collection are deallocated.
    ///
    /// If there are any active requests in the collection, then it returns `Some((idx, status))`,
    /// where `idx` is the index of the completed request in the collection and `status` is the
    /// status of the completed request. The request at `idx` will be set to None. `outstanding()`
    /// will be reduced by 1.
    /// 
    /// Returns `None` if there are no active requests. `outstanding()` is 0.
    ///
    /// # Standard section(s)
    ///
    /// 3.7.5
    pub fn wait_any(&mut self) -> Option<(i32, Status)> {
        let mut status: MPI_Status = unsafe { mem::uninitialized() };
        let result = raw::wait_any(&mut self.requests, Some(&mut status)).map(|idx| {
            self.ensure_null(idx);
            self.outstanding -= 1;
            (idx, Status::from_raw(status))
        });
        self.check_outstanding();
        result
    }

    /// `wait_any_without_status` blocks until any active request in the collection completes. It
    /// returns immediately if all requests in the collection are deallocated.
    ///
    /// If there are any active requests in the collection, then it returns `Some(idx)`, where
    /// `idx` is the index of the completed request in the collection and `status` is the status of
    /// the completed request. The request at `idx` will be set to None. `outstanding()` will be
    /// reduced by 1.
    /// 
    /// Returns `None` if there are no active requests. `outstanding()` is 0.
    ///
    /// # Standard section(s)
    ///
    /// 3.7.5
    pub fn wait_any_without_status(&mut self) -> Option<i32> {
        let result = raw::wait_any(&mut self.requests, None).map(|idx| {
            self.ensure_null(idx);
            self.outstanding -= 1;
            idx
        });
        self.check_outstanding();
        result
    }

    /// `wait_all_into` blocks until all requests in the collection are deallocated. Upon return,
    /// all requests in the collection will be deallocated. `outstanding()` will be equal to 0.
    /// `statuses` will be updated with the status for each request that is completed by
    /// `wait_all_into` where each status will match the index of the completed request. The status
    /// for deallocated entries will be set to empty.
    /// 
    /// Panics if `statuses.len()` is not >= `self.len()`.
    ///
    /// If you do not need the status of the completed requests, `wait_all_without_status` is
    /// slightly more efficient because it does not allocate memory.
    ///
    /// # Standard section(s)
    ///
    /// 3.7.5
    pub fn wait_all_into(&mut self, statuses: &mut [Status]) {
        // This code assumes that the representation of point_to_point::Status
        // is the same as ffi::MPI_Status.
        let raw_statuses = unsafe {
            slice::from_raw_parts_mut(
                statuses.as_mut_ptr() as *mut _,
                statuses.len())
        };

        raw::wait_all(&mut self.requests, Some(raw_statuses));

        self.ensure_all_null();
        self.set_outstanding(0);
    }

    /// Wait for all requests in the collection to complete.
    /// 
    /// `outstanding()` shall be equal to 0 on completion.
    /// 
    /// If you do not need the status of the completed requests,
    /// `wait_all_without_status` is slightly more efficient because it does
    /// not allocate memory.
    ///
    /// # Examples
    ///
    /// See `examples/immediate_wait_all.rs`
    /// 
    /// # Standard section(s)
    ///
    /// 3.7.5
    pub fn wait_all(&mut self) -> Vec<Status> {
        let mut statuses = vec![unsafe { mem::uninitialized() }; self.requests.len()];
        self.wait_all_into(&mut statuses[..]);
        statuses
    }

    /// Reduce the scope of a request.
    pub fn shrink_scope_to<'b, S2>(self, scope: S2) -> Request<'b, S2>
    where
        'a: 'b,
        S2: Scope<'b>,
    {
        unsafe {
            let (request, _) = self.into_raw();
            Request::from_raw(request, scope)
        }
    }

    /// `wait_all_without_status` blocks until all requests in the collection are deallocated. Upon
    /// return, all requests in the collection will be deallocated. `outstanding()` will be equal to
    /// 0.
    ///
    /// # Standard section(s)
    ///
    /// 3.7.5
    pub fn wait_all_without_status(&mut self) {
        raw::wait_all(&mut self.requests[..], None);

        self.ensure_all_null();
        self.set_outstanding(0);
    }
}

/// Guard object that waits for the completion of an operation when it is dropped
///
/// The guard can be constructed or deconstructed using the `From` and `Into` traits.
///
/// # Examples
///
/// See `examples/immediate.rs`
#[derive(Debug)]
pub struct WaitGuard<'a, S: Scope<'a> = StaticScope>(Option<Request<'a, S>>);

impl<'a, S: Scope<'a>> Drop for WaitGuard<'a, S> {
    fn drop(&mut self) {
        self.0.take().expect("invalid WaitGuard").wait();
    }
}

unsafe impl<'a, S: Scope<'a>> AsRaw for WaitGuard<'a, S> {
    type Raw = MPI_Request;
    fn as_raw(&self) -> Self::Raw {
        self.0.as_ref().expect("invalid WaitGuard").as_raw()
    }
}

impl<'a, S: Scope<'a>> From<WaitGuard<'a, S>> for Request<'a, S> {
    fn from(mut guard: WaitGuard<'a, S>) -> Self {
        guard.0.take().expect("invalid WaitGuard")
    }
}

impl<'a, S: Scope<'a>> From<Request<'a, S>> for WaitGuard<'a, S> {
    fn from(req: Request<'a, S>) -> Self {
        WaitGuard(Some(req))
    }
}

impl<'a, S: Scope<'a>> WaitGuard<'a, S> {
    fn cancel(&self) {
        if let Some(ref req) = self.0 {
            req.cancel();
        }
    }
}

/// Guard object that tries to cancel and waits for the completion of an operation when it is
/// dropped
///
/// The guard can be constructed or deconstructed using the `From` and `Into` traits.
///
/// # Examples
///
/// See `examples/immediate.rs`
#[derive(Debug)]
pub struct CancelGuard<'a, S: Scope<'a> = StaticScope>(WaitGuard<'a, S>);

impl<'a, S: Scope<'a>> Drop for CancelGuard<'a, S> {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

impl<'a, S: Scope<'a>> From<CancelGuard<'a, S>> for WaitGuard<'a, S> {
    fn from(guard: CancelGuard<'a, S>) -> Self {
        unsafe {
            let inner = ptr::read(&guard.0);
            mem::forget(guard);
            inner
        }
    }
}

impl<'a, S: Scope<'a>> From<WaitGuard<'a, S>> for CancelGuard<'a, S> {
    fn from(guard: WaitGuard<'a, S>) -> Self {
        CancelGuard(guard)
    }
}

impl<'a, S: Scope<'a>> From<Request<'a, S>> for CancelGuard<'a, S> {
    fn from(req: Request<'a, S>) -> Self {
        CancelGuard(WaitGuard::from(req))
    }
}

/// A common interface for [`LocalScope`](struct.LocalScope.html) and
/// [`StaticScope`](struct.StaticScope.html) used internally by the `request` module.
///
/// This trait is an implementation detail.  You shouldn’t have to use or implement this trait.
pub unsafe trait Scope<'a> {
    /// Registers a request with the scope.
    fn register(&self);

    /// Unregisters a request from the scope.
    ///
    /// # Safety
    /// DO NOT IMPLEMENT
    unsafe fn unregister(&self);
}

/// The scope that lasts as long as the entire execution of the program
///
/// Unlike `LocalScope<'a>`, `StaticScope` does not require any bookkeeping on the requests as every
/// request associated with a `StaticScope` can live as long as they please.
///
/// A `StaticScope` can be created simply by calling the `StaticScope` constructor.
///
/// # Invariant
///
/// For any `Request` registered with a `StaticScope`, its associated buffers must be `'static`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct StaticScope;

unsafe impl Scope<'static> for StaticScope {
    fn register(&self) {}

    unsafe fn unregister(&self) {}
}

/// A temporary scope that lasts no more than the lifetime `'a`
///
/// Use `LocalScope` for to perform requests with temporary buffers.
///
/// To obtain a `LocalScope`, use the [`scope`](fn.scope.html) function.
///
/// # Invariant
///
/// For any `Request` registered with a `LocalScope<'a>`, its associated buffers must outlive `'a`.
///
/// # Panics
///
/// When `LocalScope` is dropped, it will panic if there are any lingering `Requests` that have not
/// yet been completed.
#[derive(Debug)]
pub struct LocalScope<'a> {
    num_requests: Cell<usize>,
    phantom: PhantomData<Cell<&'a ()>>, // Cell needed to ensure 'a is invariant
}

#[cold]
fn abort_on_unhandled_request() {
    let _ = std::panic::catch_unwind(|| {
        panic!("at least one request was dropped without being completed");
    });

    // There's no way to tell MPI to release the buffers that were passed to it. Therefore
    // we must abort execution.
    std::process::abort();
}

impl<'a> Drop for LocalScope<'a> {
    fn drop(&mut self) {
        if self.num_requests.get() != 0 {
            abort_on_unhandled_request();
        }
    }
}

unsafe impl<'a, 'b> Scope<'a> for &'b LocalScope<'a> {
    fn register(&self) {
        self.num_requests.set(self.num_requests.get() + 1)
    }

    unsafe fn unregister(&self) {
        self.num_requests.set(
            self.num_requests
                .get()
                .checked_sub(1)
                .expect("unregister has been called more times than register"),
        )
    }
}

/// Used to create a [`LocalScope`](struct.LocalScope.html)
///
/// The function creates a `LocalScope` and then passes it into the given
/// closure as an argument.
///
/// For safety reasons, all variables and buffers associated with a request
/// must exist *outside* the scope with which the request is registered.
///
/// It is typically used like this:
///
/// ```
/// /* declare variables and buffers here ... */
/// mpi::request::scope(|scope| {
///     /* perform sends and/or receives using 'scope' */
/// });
/// /* at end of scope, panic if there are requests that have not yet completed */
/// ```
///
/// # Examples
///
/// See `examples/immediate.rs`
pub fn scope<'a, F, R>(f: F) -> R
where
    F: FnOnce(&LocalScope<'a>) -> R,
{
    f(&LocalScope {
        num_requests: Default::default(),
        phantom: Default::default(),
    })
}
