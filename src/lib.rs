use mpi_sys::*;
use std::ffi::{
    c_void,
    c_int,
};

#[repr(transparent)]
struct MPIType(MPI_Type);

/// Wrapping extension trait for determining MPI types from different Rust
/// types.
pub trait MPITypeExt {
    pub fn datatype() -> MPIType;
}

macro_rules! derive_mpi_type {
    ($rstype:path, $mpittype:path) => {
        impl MPITypeExt for $rstype {
            fn datatype() -> MPI_Type {
                $mpitype
            }
        }
    };
}

derive_mpi_type!(bool, MPI_C_BOOL);
derive_mpi_type!(f32, MPI_FLOAT);
derive_mpi_type!(f64, MPI_DOUBLE);

derive_mpi_type!(i8, MPI_INT8_T);
derive_mpi_type!(i16, MPI_INT16_T);
derive_mpi_type!(i32, MPI_INT32_T);
derive_mpi_type!(i64, MPI_INT64_T);

derive_mpi_type!(u8, MPI_UNT8_T);
derive_mpi_type!(u16, MPI_UNT16_T);
derive_mpi_type!(u32, MPI_UNT32_T);
derive_mpi_type!(u64, MPI_UNT64_T);

/// MPI Communication type.
#[repr(transparent)]
struct Comm(MPI_Comm);

impl Comm {
    pub const COMM_WORLD: = Self(MPI_COMM_WORLD);
    pub const COMM_NULL: = Self(MPI_COMM_NULL);
    pub const COMM_SELF: = Self(MPI_COMM_SELF);
}

impl Comm {
    #[inline]
    pub unsafe fn send<T>(&self, buf: &[T], dest: c_int, tag: c_int) -> c_int 
    where
        T: MPITypeExt,
    {
        MPI_Send(buf.as_ptr(), buf.len(), T::datatype(), det, tag, self.as_raw())
    }

    #[inline]
    pub unsafe fn recv<T>(&self, buf: &mut [T], source: c_int, tag: c_int) -> c_int
    where
        T: MPITypeExt,
    {
        let status = MaybeUnit::<MPI_Status>::uninit();
        let r = MPI_Recv(
            buf.as_mut_ptr(),
            buf.len(),
            T::datatype(),
            source,
            tag,
            self.as_raw(),
            status.as_mut_ptr(),
        );
    }
}
