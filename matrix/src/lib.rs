use num_traits::Float;
use std::{
    fmt::{Debug, Display, Formatter},
    iter::Sum,
};

///////////////////////////////////////////////////
///////////////////////////////////////////////////
//////////////////TYPES DECLARATION////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////

pub trait Scalar: Float + Copy + Display + Debug + Sum + PartialEq + Sized {}
impl<T: Float + Copy + Display + Debug + Sum> Scalar for T {}

#[derive(Clone, PartialEq, Debug)]
pub struct Vector<T> {
    data: Vec<T>,
}

#[derive(Clone, PartialEq, Default)]
pub struct Matrix<T> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
}

pub trait Tensor<T> {
    fn add(&mut self, op: &Self) -> Result<(), String>;
    fn sub(&mut self, op: &Self) -> Result<(), String>;
    fn scl(&mut self, op: T) -> Result<(), String>;
    fn check_dim_eq(&self, op: &Self) -> Result<(), String>;
}

///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////IMPLEMENTATIONS/////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////

//ex01
pub fn linear_combination<T: Scalar>(
    u: impl AsRef<[Vector<T>]>,
    coefs: impl AsRef<[T]>,
) -> Result<Vector<T>, String> {
    if coefs.as_ref().len() != u.as_ref().len() {
        return Err("Scalar list doesnt match number of vectors".into());
    }
    u.as_ref()
        .iter()
        .zip(coefs.as_ref().iter())
        .map(|(v, c)| {
            let mut r = v.clone();
            r.scl(*c)?;
            Ok(r)
        })
        .try_fold(
            None,
            |acc: Option<Vector<T>>,
             v: Result<Vector<T>, String>|
             -> Result<Option<Vector<T>>, String> {
                match acc {
                    Some(mut a) => {
                        a.add(&v?)?;
                        Ok(Some(a))
                    }
                    None => Ok(Some(v?)),
                }
            },
        )?
        .ok_or_else(|| "No vector provided".to_string())
}
//ex02
pub fn lerp<T: Scalar, S: Tensor<T>>(a: &S, b: &S, mix: T) -> Result<S, String>
where
    S: Clone,
{
    let mut res = a.clone();
    let mut diff = b.clone();
    diff.sub(a)?;
    diff.scl(mix)?;
    res.add(&diff)?;
    Ok(res)
}
//ex03
pub fn dot<T: Scalar>(a: &Vector<T>, b: &Vector<T>) -> Result<T, String> {
    let a = &a.data;
    let b = &b.data;
    if a.len() != b.len() {
        return Err("The two vectors are not of the same size".into());
    };
    Ok(a.iter().zip(b.iter()).map(|(a, b)| *a * *b).sum())
}
//ex05
pub fn angle_cos<T: Scalar>(a: &Vector<T>, b: &Vector<T>) -> Result<T, String> {
    let dot = dot(a, b)?;
    let an = a.norm_2();
    let bn = b.norm_2();
    if let (Some(an), Some(bn)) = (an, bn) {
        Ok(dot / (an * bn))
    } else {
        Err("Vectors are either empty or of size 0".into())
    }
}
//ex06
pub fn cross_product<T: Scalar>(a: &Vector<T>, b: &Vector<T>) -> Result<Vector<T>, String> {
    if a.data.len() != 3 || b.data.len() != 3 {
        return Err("vector must be of dimension 3".into());
    }
    Ok(Vector {
        data: vec![
            a.data[1] * b.data[2] - a.data[2] * b.data[1],
            a.data[2] * b.data[0] - a.data[0] * b.data[2],
            a.data[0] * b.data[1] - a.data[1] * b.data[0],
        ],
    })
}

impl<T: Scalar> Vector<T> {
    pub fn from(s: impl AsRef<[T]>) -> Self {
        Vector {
            data: s.as_ref().to_vec(),
        }
    }

    // ex04
    pub fn norm_1(&self) -> Option<T> {
        if self.data.is_empty() {
            return None;
        }
        let result = self.data.iter().copied().sum();
        (result != T::zero()).then_some(result)
    }
    pub fn norm_2(&self) -> Option<T> {
        if self.data.is_empty() {
            return None;
        }
        let result = self.data.iter().map(|x| *x * *x).sum::<T>().sqrt();
        (result != T::zero()).then_some(result)
    }
    pub fn norm_3(&self) -> Option<T> {
        if self.data.is_empty() {
            return None;
        }
        let result = self
            .data
            .iter()
            .fold(None, |acc, &x| {
                let abs = x.abs();
                match acc {
                    Some(max) if max >= abs => Some(max),
                    _ => Some(abs),
                }
            })
            .unwrap();
        (result != T::zero()).then_some(result)
    }
}
impl<T: Scalar> Tensor<T> for Vector<T> {
    fn check_dim_eq(&self, v: &Vector<T>) -> Result<(), String> {
        if self.data.len() != v.data.len() {
            return Err("Vector size mismatch".into());
        };
        Ok(())
    }
    fn add(&mut self, v: &Vector<T>) -> Result<(), String> {
        self.check_dim_eq(v)?;
        self.data
            .iter_mut()
            .zip(v.data.iter())
            .for_each(|(a, b)| *a = *a + *b);
        Ok(())
    }

    fn sub(&mut self, v: &Vector<T>) -> Result<(), String> {
        self.check_dim_eq(v)?;
        self.data
            .iter_mut()
            .zip(v.data.iter())
            .for_each(|(a, b)| *a = *a - *b);
        Ok(())
    }
    fn scl(&mut self, s: T) -> Result<(), String> {
        self.data.iter_mut().for_each(|a| *a = *a * s);
        Ok(())
    }
}

impl<T: Scalar> Matrix<T> {
    pub fn identity(c: usize, r: usize) -> Self {
        let mut vec: Vec<T> = vec![T::zero(); c * r];
        for c in 0..c {
            vec[c * r + c] = T::one();
        }

        Matrix {
            data: vec,
            rows: r,
            cols: c,
        }
    }
    pub fn from(s: impl AsRef<[T]>, c: usize, r: usize) -> Self {
        assert_eq!(
            s.as_ref().len(),
            r * c,
            "Values provides doesnt match dimensions"
        );
        Matrix {
            data: s.as_ref().to_vec(),
            rows: r,
            cols: c,
        }
    }
    //ex07
    pub fn mul_vec(&self, vec: &Vector<T>) -> Result<Vector<T>, String> {
        if self.cols != vec.data.len() {
            return Err("Matrix size doesnt match vector".into());
        }
        let mut res = Vec::<T>::new();
        for r in 0..self.rows {
            let mut crr = T::zero();
            for c in 0..self.cols {
                crr = crr + self.data[r * self.cols + c] * vec.data[c];
            }
            res.push(crr);
        }
        Ok(Vector { data: res })
    }
    pub fn mul_mat(&self, mat: &Matrix<T>) -> Result<Matrix<T>, String> {
        if self.cols != mat.rows {
            return Err("Matrixes dimensions doesnt match".into());
        }
        let mut res = Vec::<T>::new();
        for r in 0..self.rows {
            for c in 0..mat.cols {
                let mut crr = T::zero();
                for k in 0..self.cols {
                    crr = crr + self.data[r * self.cols + k] * mat.data[k * mat.cols + c];
                }
                res.push(crr);
            }
        }
        Ok(Matrix {
            data: res,
            rows: self.rows,
            cols: mat.cols,
        })
    }
    //ex08
    pub fn trace(&self) -> Result<T, String> {
        if self.cols != self.rows {
            return Err("Trace require a square matrix".into());
        }
        let mut sum = T::zero();
        for i in 0..self.rows {
            sum = sum + self.data[i * self.cols + i];
        }
        Ok(sum)
    }
    //ex09
    pub fn transpose(&self) -> Matrix<T> {
        let mut new = vec![T::zero(); self.cols * self.rows];
        for r in 0..self.rows {
            for c in 0..self.cols {
                new[r + c * self.rows] = self.data[c + r * self.cols];
            }
        }
        Matrix {
            data: new,
            cols: self.rows,
            rows: self.cols,
        }
    }
    //ex10
    fn swap_rows(&mut self, idx: (usize, usize)) -> Result<(), String> {
        if idx.0 >= self.rows || idx.1 >= self.rows {
            return Err(format!("indexes is out of range: {:?} for mat{self}", idx,).to_string());
        }
        if idx.0 == idx.1 {
            return Ok(());
        }
        let (r1, r2) = if idx.0 < idx.1 {
            (idx.0, idx.1)
        } else {
            (idx.1, idx.0)
        };
        let start1 = r1 * self.cols;
        let start2 = r2 * self.cols;
        let (left, right) = self.data.split_at_mut(start2);
        let row1 = &mut left[start1..start1 + self.cols];
        let row2 = &mut right[..self.cols];
        row1.swap_with_slice(row2);
        Ok(())
    }

    fn scale_row(&mut self, idx: usize, scaler: T) -> Result<(), String> {
        if idx >= self.rows {
            return Err(format!("indexes is out of range: {:?} for mat{self}", idx,).to_string());
        } else if scaler == T::zero() {
            return Err("Scaling by 0".to_string());
        }
        for n in 0..self.cols {
            self.data[idx * self.cols + n] = self.data[idx * self.cols + n] * scaler;
        }
        Ok(())
    }

    fn add_row(&mut self, idx: (usize, usize), factor: T) -> Result<(), String> {
        if idx.0 >= self.rows || idx.1 >= self.rows {
            return Err(format!("indexes is out of range: {:?} for mat{self}", idx,).to_string());
        }
        for n in 0..self.cols {
            self.data[idx.0 * self.cols + n] =
                self.data[idx.0 * self.cols + n] + self.data[idx.1 * self.cols + n] * factor;
        }
        Ok(())
    }

    pub fn row_echelon(&mut self) -> i32 {
        macro_rules! at {
            ($r: expr, $c: expr) => {
                self.data[$r * self.cols + $c]
            };
        }
        let mut r_start = 0;
        let mut swap_count = 0;
        for c in 0..self.cols {
            let mut pivot: Option<usize> = None;
            for r in r_start..self.rows {
                if at!(r, c) == T::zero() {
                    if r <= self.rows - 2 && pivot.is_none() {
                        self.swap_rows((r, r + 1)).expect("swap rows error");
                        swap_count += 1;
                    }
                    continue;
                }
                if let Some(pivot) = pivot {
                    self.add_row((r, pivot), -at!(r, c) / at!(pivot, c))
                        .expect("add row error");
                } else {
                    pivot = Some(r);
                    r_start += 1;
                }
            }
        }
        swap_count
    }
    pub fn reduced_row_echelon(&mut self) -> i32 {
        macro_rules! at {
            ($r: expr, $c: expr) => {
                self.data[$r * self.cols + $c]
            };
        }
        let mut r_start = 0;
        let mut swap_count = 0;
        for c in 0..self.cols {
            let mut pivot: Option<usize> = None;
            for r in r_start..self.rows {
                if at!(r, c) == T::zero() {
                    if r <= self.rows - 2 && pivot.is_none() {
                        if at!(r + 1, c) != T::zero() {
                            self.scale_row(r + 1, T::one() / at!(r + 1, c))
                                .expect("scale row error");
                        }
                        self.swap_rows((r, r + 1)).expect("swap rows error");
                        swap_count += 1;
                    }
                    continue;
                }
                if let Some(pivot) = pivot {
                    self.add_row((r, pivot), -at!(r, c)).expect("add row error");
                } else {
                    self.scale_row(r, T::one() / at!(r, c))
                        .expect("scale row error");
                    pivot = Some(r);
                    r_start += 1;
                }
            }
        }
        swap_count
    }
    pub fn determinant(&self) -> Result<f32, String> {
        if self.cols != self.rows {
            return Err("Ta pas capte, c pas carré".to_string());
        }
        let mut mat = self.clone();
        let swap = mat.row_echelon();
        let mut result = if swap % 2 != 0 { -1.0_f32 } else { 1.0_f32 };
        for i in 0..self.cols {
            if mat.data[i * mat.cols + i] == T::zero() {
                return Ok(0.);
            }
            result = result * mat.data[i * mat.cols + i].to_f32().unwrap();
        }
        Ok(result)
    }
    pub fn augment(&mut self, augment: Matrix<T>) -> Result<(), String> {
        if self.rows != augment.rows {
            return Err("Both matrix must have the same row numbers to be augmented".to_string());
        }
        let mut data: Vec<T> = Vec::with_capacity((self.cols + augment.cols) * self.rows);
        let cols = self.cols + augment.cols;
        for r in 0..self.rows {
            for c in 0..self.cols {
                data.push(self.data[r * self.cols + c]);
            }
            for c in 0..augment.cols {
                data.push(augment.data[r * augment.cols + c]);
            }
        }
        self.data = data;
        self.cols = cols;
        Ok(())
    }
    pub fn plu(&self) -> Result<(Matrix<T>, Matrix<T>, Matrix<T>), String> {
        if self.cols != self.rows {
            return Err("matrice isnt square".to_string());
        }
        let mut p = Matrix::identity(self.cols, self.rows);
        let mut l = Matrix::identity(self.cols, self.rows);
        let mut u = self.clone();

        macro_rules! at {
            ($r: expr, $c: expr) => {
                u.data[$r * u.cols + $c]
            };
        }
        let mut r_start = 0;
        for c in 0..u.cols {
            let mut pivot: Option<usize> = None;
            for r in r_start..u.rows {
                if at!(r, c) == T::zero() {
                    if r <= u.rows - 2 && pivot.is_none() {
                        u.swap_rows((r, r + 1)).expect("swap rows error");
                        //Register swap in pivot Matrix
                        l.swap_rows((r, r + 1)).expect("swap rows error");
                    }
                    continue;
                }
                if let Some(pivot) = pivot {
                    let scaler = at!(r, c) / at!(pivot, c);
                    u.add_row((r, pivot), -scaler).expect("add row error");
                    // register scale in lower matrix
                    l.data[r * u.cols + c] = scaler;
                } else {
                    pivot = Some(r);
                    r_start += 1;
                }
            }
        }
        Ok((p, l, u))
    }

    pub fn solve(&self, &result: Matrix<T>) -> Result<Matrix<T>, String> {
        if self.cols != result.rows {
            return Err("Matrix dimensions does not match expected result".to_string());
        }
        let mut solution = Matrix::from(vec![0.; self.rows * self.cols], self.cols, self.rows);

        for c in 0..solution.cols {
            for r in 0..solution.rows {
                
            }
        }
        Ok(solution)
    }

    pub fn inverse(&mut self) -> Result<(), String> {
        if self.rows != self.cols {
            return Err("Matrix must be squared to be inversed.".to_string());
        }
        let (p, l, u) = self.plu().expect("plu decomposition error");
        let ux = l.solve(p.transpose());

        Ok(())
    }
}

impl<T: Scalar> Tensor<T> for Matrix<T> {
    fn check_dim_eq(&self, m: &Matrix<T>) -> Result<(), String> {
        if self.cols != m.cols {
            return Err(format!("Cols mismatch {} : {}", self.cols, m.cols));
        }
        if self.rows != m.rows {
            return Err(format!("Rows mismatch {} : {}", self.rows, m.rows));
        }
        Ok(())
    }
    fn add(&mut self, m: &Matrix<T>) -> Result<(), String> {
        self.check_dim_eq(m)?;
        self.data
            .iter_mut()
            .zip(m.data.iter())
            .for_each(|(a, b)| *a = *a + *b);
        Ok(())
    }
    fn sub(&mut self, m: &Matrix<T>) -> Result<(), String> {
        self.check_dim_eq(m)?;
        self.data
            .iter_mut()
            .zip(m.data.iter())
            .for_each(|(a, b)| *a = *a - *b);
        Ok(())
    }
    fn scl(&mut self, s: T) -> Result<(), String> {
        self.data.iter_mut().for_each(|a| *a = *a * s);
        Ok(())
    }
}

impl<T: Scalar> Display for Vector<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.data)
    }
}
impl<T: Scalar> Debug for Matrix<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "\x1B[s")?;
        let width = self
            .data
            .iter()
            .map(|x| format!("{}", x).len())
            .max()
            .unwrap_or(0);
        for r in 0..self.rows {
            write!(f, "\x1B[u")?;
            if r > 0 {
                write!(f, "\x1B[{}B", r)?;
            }
            write!(f, "[")?;
            for c in 0..self.cols {
                let sep = match c {
                    c if c != self.cols - 1 => ",",
                    c if c == self.cols - 1 && r != self.rows - 1 => "]\n",
                    _ => "]",
                };
                write!(
                    f,
                    "{:>width$}{}",
                    &self.data[r * self.cols + c],
                    sep,
                    width = width
                )?;
            }
        }
        Ok(())
    }
}
impl<T: Scalar> Display for Matrix<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "\x1B[s")?;
        let width = self
            .data
            .iter()
            .map(|x| format!("{}", x).len())
            .max()
            .unwrap_or(0);
        for r in 0..self.rows {
            write!(f, "\x1B[u")?;
            if r > 0 {
                write!(f, "\x1B[{}B", r)?;
            }
            write!(f, "[")?;
            for c in 0..self.cols {
                let sep = match c {
                    c if c != self.cols - 1 => ",",
                    c if c == self.cols - 1 && r != self.rows - 1 => "]\n",
                    _ => "]",
                };
                write!(
                    f,
                    "{:>width$}{}",
                    &self.data[r * self.cols + c],
                    sep,
                    width = width
                )?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[rustfmt::skip]
mod tests {
    use super::*;

    #[test]
    fn ex00_valid_constructors() {
        let empty = Vector::<f32>::from([]);
        assert!(empty.data.is_empty());
        let vec = Vector::from([1., 2., 3.]);
        assert_eq!(
            vec,
            Vector {
                data: vec!(1., 2., 3.)
            }
        );
        let vec = Vector::from([1., 2., 3.]);
        assert_eq!(
            vec,
            Vector {
                data: vec!(1., 2., 3.)
            }
        );
        let vec = Vector::from([0.; 10]);
        assert_eq!(
            vec,
            Vector {
                data: vec!(0., 0., 0., 0., 0., 0., 0., 0., 0., 0.)
            }
        );
        let empty = Matrix::<f32>::from([], 0, 0);
        assert!(empty.data.is_empty());
        assert!(empty.cols == 0 && empty.rows == 0);
        let mat = Matrix::from([1., 2., 3.], 1, 3);
        assert_eq!(
            mat,
            Matrix {
                data: vec!(1., 2., 3.),
                cols: 1,
                rows: 3,
            }
        );
        let mat = Matrix::from([1., 2., 3.], 3, 1);
        assert_eq!(
            mat,
            Matrix {
                data: vec!(1., 2., 3.),
                cols: 3,
                rows: 1,
            }
        );
        let mat = Matrix::from([0.; 10], 2, 5);
        assert_eq!(
            mat,
            Matrix {
                data: vec!(0.; 10),
                cols: 2,
                rows: 5,
            }
        );
    }
    #[test]
    #[should_panic]
    fn ex00_invalid_constructor() {
        Matrix::from([0.; 10], 20, 5);
    }

    #[test]
    fn ex00_test_add() {
        let a = Vector::from([1.; 1]);
        let b = Vector::from([2.; 1]);
        let mut r = a.clone();
        assert!(r.add(&b).is_ok());
        assert_eq!(r, Vector { data: vec!(3.) });
        let a = Matrix::from([1.; 10], 2, 5);
        let b = Matrix::from([2.; 10], 2, 5);
        let mut r = a.clone();
        assert!(r.add(&b).is_ok());
        assert_eq!(
            r,
            Matrix {
                data: vec!(3.; 10),
                cols: 2,
                rows: 5
            }
        );
        let a = Vector::from([1.; 10]);
        let b = Vector::from([2.; 15]);
        let mut r = a.clone();
        assert!(r.add(&b).is_err());
        let a = Matrix::from([1.; 10], 2, 5);
        let b = Matrix::from([2.; 10], 5, 2);
        let mut r = a.clone();
        assert!(r.add(&b).is_err());
    }
    #[test]
    fn ex00_valid_sum() {
        let a = Vector::from([1.; 10]);
        let b = Vector::from([2.; 10]);
        let mut r = a.clone();
        assert!(r.sub(&b).is_ok());
        assert_eq!(
            r,
            Vector {
                data: vec!(-1.; 10)
            }
        );
        let a = Matrix::from([1.; 10], 2, 5);
        let b = Matrix::from([2.; 10], 2, 5);
        let mut r = a.clone();
        assert!(r.sub(&b).is_ok());
        assert_eq!(
            r,
            Matrix {
                data: vec!(-1.; 10),
                cols: 2,
                rows: 5,
            }
        );
        let a = Vector::from([1.; 10]);
        let b = Vector::from([2.; 15]);
        let mut r = a.clone();

        assert!(r.sub(&b).is_err());
        let a = Matrix::from([1.; 10], 2, 5);
        let b = Matrix::from([2.; 10], 5, 2);
        let mut r = a.clone();
        assert!(r.sub(&b).is_err());
    }
    #[test]
    fn ex00_valid_scl() {
        let scaler: f32 = 42.;
        let mut a = Vector::from([1.; 10]);
        assert!(a.scl(scaler).is_ok());
        assert_eq!(
            a,
            Vector {
                data: vec!(42.; 10)
            }
        );
        let mut a = Matrix::from([1.; 9], 3, 3);
        assert!(a.scl(scaler).is_ok());
        assert_eq!(
            a,
            Matrix {
                data: vec!(42.; 9),
                cols: 3,
                rows: 3,
            }
        );
    }

    #[test]
    fn ex01_valid_linear() {
        let e1 = Vector::from([1., 0., 0.]);
        let e2 = Vector::from([0., 1., 0.]);
        let e3 = Vector::from([0., 0., 1.]);
        let v1 = Vector::from([1., 2., 3.]);
        let v2 = Vector::from([0., 10., -100.]);
        assert_eq!(
            Vector {
                data: vec!(10., -2., 0.5)
            },
            linear_combination([e1, e2, e3], [10., -2., 0.5]).unwrap()
        );
        assert_eq!(
            Vector {
                data: vec!(10., 0., 230.)
            },
            linear_combination([v1, v2], [10., -2.]).unwrap()
        );
        let e1 = Vector::from([1., 0., 0., 4.]); // Vectors of different shapes
        let e2 = Vector::from([0., 1., 0.]);
        let e3 = Vector::from([0., 0., 1.]);
        assert!(linear_combination([e1, e2, e3], [10., -2., 0.5]).is_err());

        let e1 = Vector::from([1., 0., 0.]); // Scalar list doesnt match number of vectors
        let e2 = Vector::from([0., 1., 0.]);
        let e3 = Vector::from([0., 0., 1.]);
        assert!(linear_combination([e1, e2, e3], [10., -2., 0.5, 3.]).is_err());
    }

    #[test]
    fn ex_02_lerp() {
        let a = Vector::from([1.; 2]);
        let b = Vector::from([-1.; 2]);
        let mut mix = 0.5;
        let res = lerp(&a, &b, mix).unwrap();
        assert_eq!(res, Vector { data: vec!(0.; 2) });
        mix = 0.;
        let res = lerp(&a, &b, mix).unwrap();
        assert_eq!(res, a);
        mix = 1.;
        let res = lerp(&a, &b, mix).unwrap();
        assert_eq!(res, b);

        let c = Matrix::from([1.; 9], 3, 3);
        let d = Matrix::from([-1.; 9], 3, 3);
        let mut mix = 0.5;
        let res = lerp(&c, &d, mix).unwrap();
        assert_eq!(
            res,
            Matrix {
                data: vec!(0.; 9),
                cols: 3,
                rows: 3
            }
        );
        mix = 0.;
        let res = lerp(&c, &d, mix).unwrap();
        assert_eq!(
            res,
            Matrix {
                data: vec!(1.; 9),
                cols: 3,
                rows: 3
            }
        );
        mix = 1.;
        let res = lerp(&c, &d, mix).unwrap();
        assert_eq!(
            res,
            Matrix {
                data: vec!(-1.; 9),
                cols: 3,
                rows: 3
            }
        );
        let a2 = Vector::from([3.; 6]);
        assert!(lerp(&a, &a2, mix).is_err());
        let c2 = Matrix::from([3.; 2], 1, 2);
        assert!(lerp(&c, &c2, mix).is_err());
    }
    #[test]
    fn ex03() {
        let vn = Vector::<f32>::from([]);
        assert_eq!(dot(&vn, &vn), Ok(0.));
        let v1 = Vector::from([0., 1.]);
        let v2 = Vector::from([1., 0.]);
        assert_eq!(dot(&v1, &v2), Ok(0.));
        let v1 = Vector::from([1., 1.]);
        let v2 = Vector::from([-1., 1.]);
        assert_eq!(dot(&v1, &v2), Ok(0.));
        let v1 = Vector::from([1., 1.]);
        let v2 = Vector::from([1., 1.]);
        assert_eq!(dot(&v1, &v2), Ok(2.));
        let v1 = Vector::from([-1., 6.]);
        let v2 = Vector::from([3., 2.]);
        assert_eq!(dot(&v1, &v2), Ok(9.));
    }
    #[test]
    fn ex04_norms() {
        let vx = Vector::from([3., 0., 0.]);
        let vy = Vector::from([0., 3., 0.]);
        let vz = Vector::from([0., 0., 3.]);
        let empty = Vector::<f32>::from([]);
        let zero = Vector::<f32>::from([0.; 3]);
        assert!(empty.norm_1().is_none());
        assert!(empty.norm_2().is_none());
        assert!(empty.norm_3().is_none());
        assert!(zero.norm_1().is_none());
        assert!(zero.norm_2().is_none());
        assert!(zero.norm_3().is_none());
        assert_eq!(vx.norm_1(), Some(3.0));
        assert_eq!(vy.norm_1(), Some(3.0));
        assert_eq!(vz.norm_1(), Some(3.0));
        assert_eq!(vx.norm_2(), Some(3.0));
        assert_eq!(vy.norm_2(), Some(3.0));
        assert_eq!(vz.norm_2(), Some(3.0));
        assert_eq!(vx.norm_3(), Some(3.0));
        assert_eq!(vy.norm_3(), Some(3.0));
        assert_eq!(vz.norm_3(), Some(3.0));
        let vecteureu = Vector::from([-20., 10., -3.]);
        assert_eq!(vecteureu.norm_3(), Some(20.));
        let mut v = vx.clone();
        assert!(v.add(&vy).is_ok());
        assert!(v.add(&vz).is_ok());
        assert_eq!(v.norm_1(), Some(9.0));
        let fix_norm = Vector::from([80198051.0; 3]);
        assert_eq!(fix_norm.norm_2(), Some(138907099.0));
    }
    #[test]
    fn ex05_cos() {
        let vnull = Vector::from([0.; 3]);
        let vx = Vector::from([3., 0., 0.]);
        let vy = Vector::from([0., 3., 0.]);
        let mut vy_neg = vy.clone();
        vy_neg.scl(-1.).expect("Scaling error");
        assert!(angle_cos(&vnull, &vx).is_err());
        assert_eq!(angle_cos(&vx, &vx), Ok(1.));
        assert_eq!(angle_cos(&vy, &vy_neg), Ok(-1.));
    }
    #[test]
    fn ex06_cross() {
        let a = Vector::from([1., 0., 0.]);
        let b = Vector::from([0., 1., 0.]);
        let c = Vector::from([0., 0., 1.]);
        assert_eq!(cross_product(&a, &b).unwrap(), c);
        let a = Vector::from([1., 0.]);
        let b = Vector::from([0., 1.]);
        assert!(cross_product(&a, &b).is_err());
    }

    #[test]
    fn ex07_mul_vec() {
        let mat = Matrix::from([1.; 9], 3, 3);
        let vec = Vector::from([3.; 2]);
        assert!(mat.mul_vec(&vec).is_err());
        let vec = Vector::from([3.; 3]);
        assert_eq!(mat.mul_vec(&vec).unwrap(), Vector { data: vec![9.; 3] });
        let mat = Matrix::from([1., 2., 3., 4.], 2, 2); // rows=2, cols=2 => [[1,2],[3,4]]
        let vec = Vector::from([5., 6.]);
        assert_eq!(
            mat.mul_vec(&vec).unwrap(),
            Vector {
                data: vec![17., 39.]
            }
        );
        let mat = Matrix::from([1., 2., 3.], 3, 1); // rows=1, cols=3 => [[1,2,3]]
        let vec = Vector::from([4., 5., 6.]);
        assert_eq!(mat.mul_vec(&vec).unwrap(), Vector { data: vec![32.] });

        let a = Matrix::from([1., 2., 3., 4., 5., 6., 7., 8., 9.], 3, 3);
        let id = Matrix::from([1., 0., 0., 0., 1., 0., 0., 0., 1.], 3, 3);
        assert_eq!(a.mul_mat(&id).unwrap(), a.clone());
        assert_eq!(id.mul_mat(&a).unwrap(), a);
        let a = Matrix::from([1., 2., 3., 4., 5., 6.], 3, 2); // rows=2 cols=3 => [[1,2,3],[4,5,6]]
        let b = Matrix::from([7., 8., 9., 10., 11., 12.], 2, 3); // rows=3 cols=2 => [[7,8],[9,10],[11,12]]
        let c = a.mul_mat(&b).unwrap();
        assert_eq!(
            c,
            Matrix {
                data: vec!(58., 64., 139., 154.),
                rows: 2,
                cols: 2
            }
        );
        let a = Matrix::from([1.; 4], 2, 2);
        let b = Matrix::from([1.; 6], 2, 3);
        assert!(a.mul_mat(&b).is_err());
    }
    #[test]
    fn ex08_trace() {
        // simple 2x2 trace
        let a = Matrix::from([1., 2., 3., 4.], 2, 2); // rows=2, cols=2 => [[1,2],[3,4]]
        assert_eq!(a.trace().unwrap(), 5.);

        // non-square matrix should error
        let b = Matrix::from([1.; 6], 3, 2); // rows=2, cols=3 -> not square
        assert!(b.trace().is_err());
    }
    #[test]
    fn ex09_transpose() {
        let mat = Matrix::from(vec![1., 2., 3., 4., 5., 6., 7., 8.], 4, 2);
        assert_eq!(
            mat.transpose(),
            Matrix {
                data: vec![1., 5., 2., 6., 3., 7., 4., 8.],
                cols: 2,
                rows: 4,
            }
        );
        let mat = Matrix::from(vec![1.], 1, 1);
        assert_eq!(mat.transpose(), mat);
    }
    #[test]
    fn ex10_row_operations() {
        // Base matrix (3x3):
        // [1,2,3]
        // [4,5,6]
        // [7,8,9]
        let base = Matrix::from([1., 2., 3., 4., 5., 6., 7., 8., 9.], 3, 3);

        // swap_rows valid
        let mut m = base.clone();
        assert!(m.swap_rows((0, 2)).is_ok());
        assert_eq!(
            m,
            Matrix {
                data: vec![7., 8., 9., 4., 5., 6., 1., 2., 3.],
                cols: 3,
                rows: 3
            }
        );

        // swap_rows invalid index
        let mut m = base.clone();
        assert!(m.swap_rows((0, 3)).is_err());

        // scale_row valid
        let mut m = base.clone();
        assert!(m.scale_row(1, 2.).is_ok()); // row [4,5,6] -> [8,10,12]
        assert_eq!(
            m,
            Matrix {
                data: vec![1., 2., 3., 8., 10., 12., 7., 8., 9.],
                cols: 3,
                rows: 3
            }
        );

        // scale_row invalid index
        let mut m = base.clone();
        assert!(m.scale_row(3, 2.).is_err());

        // scale_row by zero should error
        let mut m = base.clone();
        assert!(m.scale_row(1, 0.).is_err());

        // add_row valid: row0 += row2 => [1,2,3] + [7,8,9] = [8,10,12]
        let mut m = base.clone();
        assert!(m.add_row((0, 2), 1.).is_ok());
        assert_eq!(
            m,
            Matrix {
                data: vec![8., 10., 12., 4., 5., 6., 7., 8., 9.],
                cols: 3,
                rows: 3
            }
        );

        // add_row invalid index
        let mut m = base;
        assert!(m.add_row((1, 3), 1.).is_err());
    }
    #[test]
    fn ex10_row_echelon() {
        // IDENTITY MATRIX - should remain unchanged for row_echelon
        let mut m = Matrix {
            data: vec![1., 0., 0., 0., 01., 00., 0., 0., 1.],
            cols: 3,
            rows: 3,
        };
        let expected = m.clone();
        m.row_echelon();
        assert_eq!(m, expected);

        // Test reduced_row_echelon on identity matrix
        let mut m = Matrix {
            data: vec![1., 0., 0., 0., 01., 00., 0., 0., 1.],
            cols: 3,
            rows: 3,
        };
        let expected = m.clone();
        m.reduced_row_echelon();
        assert_eq!(m, expected);

        // SAME BUT TRANSLATE BY 1 COLUMN - row_echelon form
        let mut m = Matrix {
            data: vec![
                0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
            ],
            cols: 4,
            rows: 4,
        };
        let expected = Matrix {
            data: vec![
                0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
            ],
            cols: 4,
            rows: 4,
        };
        m.row_echelon();
        assert_eq!(m, expected);

        // Test reduced_row_echelon on same matrix
        let mut m = Matrix {
            data: vec![
                0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
            ],
            cols: 4,
            rows: 4,
        };
        let expected = Matrix {
            data: vec![
                0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
            ],
            cols: 4,
            rows: 4,
        };
        m.reduced_row_echelon();
        assert_eq!(m, expected);

        // empty column - row_echelon form (keeps pivot as 5)
        let mut m = Matrix {
            data: vec![
                0., 0., 0., 0., 0., 5., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
            ],
            cols: 4,
            rows: 4,
        };
        let expected = Matrix {
            data: vec![
                0., 5., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
            ],
            cols: 4,
            rows: 4,
        };
        m.row_echelon();
        assert_eq!(m, expected);

        // Test reduced_row_echelon on same matrix
        let mut m = Matrix {
            data: vec![
                0., 0., 0., 0., 0., 5., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
            ],
            cols: 4,
            rows: 4,
        };
        let expected = Matrix {
            data: vec![
                0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
            ],
            cols: 4,
            rows: 4,
        };
        m.reduced_row_echelon();
        assert_eq!(m, expected);
    }

    #[test]
    fn ex11_determinant() {
        // Test identity matrix (determinant = 1)
        let mat = Matrix::from(
            vec![
                1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
            ],
            4,
            4,
        );
        assert_eq!(mat.determinant(), Ok(1.0));

        // Test singular matrix with zero row (determinant = 0)
        let mat = Matrix::from(
            vec![
                1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
            ],
            4,
            4,
        );
        assert_eq!(mat.determinant(), Ok(0.0));

        // Test matrix with specific values
        let mat = Matrix::from(
            vec![
                0., 0., 0., 0., 0., 5., 0., 0., 0., 0., 0., 0., 0., 0., 0., -11.,
            ],
            4,
            4,
        );
        assert_eq!(mat.determinant(), Ok(0.0));

        // Test 3x3 matrix
        let mat = Matrix::from(vec![78., 88., -23., 21., 11., 5., 652., 0., 13.], 3, 3);
        let expected_det = 78.0 * (11.0 * 13.0 - 5.0 * 0.0) - 88.0 * (21.0 * 13.0 - 5.0 * 652.0)
            + (-23.0) * (21.0 * 0.0 - 11.0 * 652.0);
        assert_eq!(mat.determinant(), Ok(expected_det));

        // Test non-square matrix (should return error)
        let mat = Matrix::from(
            vec![
                1., 2., 3., 4., 28., 4., 2.5, 20., 4., -4., 8., 5., 1., 4., 17.,
            ],
            5,
            3,
        );
        assert!(mat.determinant().is_err());
    }
    #[test]
    fn ex12_invert() {
        //TEST UTILS
        {


            // test identity
            assert_eq!(
                Matrix::identity(2, 2),
                Matrix {
                    data: vec![
                        1., 0.,
                        0., 1.,
                    ],
                    cols: 2,
                    rows: 2,
                }
            );
            assert_eq!(
                Matrix::identity(3, 3),
                Matrix {
                    data: vec![
                        1., 0., 0.,
                        0., 1., 0.,
                        0., 0., 1.,
                    ],
                    cols: 3,
                    rows: 3,
                }
            );
            // test augment matrix
            let mut mat1 =  Matrix {
                            data: vec![
                                1., 0., 0.,
                                0., 1., 0.,
                                0., 0., 1.,
                            ],
                            cols: 3,
                            rows: 3,
                        };
            let mat2 = Matrix {
                data: vec![
                    1., 0.,
                    0., 1.,
                    0., 0.,
                ],
                cols: 2,
                rows: 3,
            };
            let wrongmat = Matrix {
                data: vec![
                    1., 0.,
                    0., 1.,
                ],
                cols: 2,
                rows: 2,
            };
            assert!(mat1.augment(wrongmat).is_err());
    }
        // TEST PLU decomposition
        {
            let m = Matrix::from(vec![
                0., 0., 0., 0.,
                0., 0., 0., 0.,
                0., 0., 0., 0.,
            ], 4,3);
            assert!(m.plu().is_err());
        }
        {
            let m = Matrix::from(vec![
                0., 0., 0., 0.,
                0., 0., 0., 0.,
                0., 0., 0., 0.,
                0., 0., 0., 0.,
            ], 4,4);
            let (p,l,u) = m.plu().expect("plu decomposition error");
            dbg!(&p, &l, &u);
            let mut res = p.mul_mat(&l).expect("mul error");
            res = res.mul_mat(&u).expect("mul error");
            assert_eq!(m, res);
        }
        {
            let m = Matrix::from(vec![
                2., 3., 6., 4.,
                2., 3., 6., 4.,
                2., 3., 6., 4.,
                2., 3., 6., 4.,
            ], 4,4);
            let (p,l,u) = m.plu().expect("plu decomposition error");
            dbg!(&p, &l, &u);
            let mut res = p.mul_mat(&l).expect("mul error");
            res = res.mul_mat(&u).expect("mul error");
            assert_eq!(m, res);
        }


    }
}
