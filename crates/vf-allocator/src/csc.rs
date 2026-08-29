//! CSC (Compressed Sparse Column) matrix conversion helper for OSQP.

use osqp::CscMatrix;
use std::borrow::Cow;

/// Converts any nalgebra matrix (dense, static, or dynamic) to an OSQP CscMatrix.
pub fn convert_to_csc<R, C, S>(matrix: &nalgebra::Matrix<f64, R, C, S>) -> CscMatrix<'static>
where
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    S: nalgebra::RawStorage<f64, R, C>,
{
    let nrows = matrix.nrows();
    let ncols = matrix.ncols();

    let mut indptr = Vec::with_capacity(ncols + 1);
    let mut indices = Vec::new();
    let mut data = Vec::new();

    indptr.push(0);
    for col in 0..ncols {
        for row in 0..nrows {
            let val = matrix[(row, col)];
            if val.abs() > 1e-12 {
                indices.push(row);
                data.push(val);
            }
        }
        indptr.push(data.len());
    }

    CscMatrix {
        nrows,
        ncols,
        indptr: Cow::Owned(indptr),
        indices: Cow::Owned(indices),
        data: Cow::Owned(data),
    }
}

/// Converts the upper triangular part of any square nalgebra matrix to an OSQP CscMatrix.
pub fn convert_to_csc_upper_tri<R, C, S>(
    matrix: &nalgebra::Matrix<f64, R, C, S>,
) -> CscMatrix<'static>
where
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    S: nalgebra::RawStorage<f64, R, C>,
{
    let nrows = matrix.nrows();
    let ncols = matrix.ncols();
    assert_eq!(nrows, ncols, "Matrix must be square");

    let mut indptr = Vec::with_capacity(ncols + 1);
    let mut indices = Vec::new();
    let mut data = Vec::new();

    indptr.push(0);
    for col in 0..ncols {
        for row in 0..=col {
            let val = matrix[(row, col)];
            if val.abs() > 1e-12 {
                indices.push(row);
                data.push(val);
            }
        }
        indptr.push(data.len());
    }

    CscMatrix {
        nrows,
        ncols,
        indptr: Cow::Owned(indptr),
        indices: Cow::Owned(indices),
        data: Cow::Owned(data),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Matrix3;

    #[test]
    fn test_dense_to_csc() {
        let mat = Matrix3::new(1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 4.0, 5.0, 0.0);

        let csc = convert_to_csc(&mat);

        assert_eq!(csc.nrows, 3);
        assert_eq!(csc.ncols, 3);
        assert_eq!(csc.indptr.as_ref(), &[0, 2, 3, 5]);
        assert_eq!(csc.indices.as_ref(), &[0, 2, 2, 0, 1]);
        assert_eq!(csc.data.as_ref(), &[1.0, 4.0, 5.0, 2.0, 3.0]);
    }

    #[test]
    fn test_dense_to_csc_upper_tri() {
        let mat = Matrix3::new(1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 4.0, 5.0, 0.0);

        let csc = convert_to_csc_upper_tri(&mat);

        assert_eq!(csc.nrows, 3);
        assert_eq!(csc.ncols, 3);
        assert_eq!(csc.indptr.as_ref(), &[0, 1, 1, 3]);
        assert_eq!(csc.indices.as_ref(), &[0, 0, 1]);
        assert_eq!(csc.data.as_ref(), &[1.0, 2.0, 3.0]);
    }
}
