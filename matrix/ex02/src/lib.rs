use std::{
    fmt::{Debug, Display, Formatter, Result},
    ops::{Add, Mul, Sub},
};

///////////////////////////////////////////////////
///////////////////////////////////////////////////
//////////////////TYPES DECLARATION////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////

pub trait Scalar:
    Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Copy + Display + Debug {}

pub trait Tensor<T: Scalar> {
    fn add(&mut self, v: &Self);
    fn sub(&mut self, v: &Self);
    fn scl(&mut self, s: T);
}

#[derive(Clone, PartialEq, Debug)]
pub struct Vector<T> {
    data: Vec<T>,
}
#[derive(Clone, PartialEq, Debug)]
pub struct Matrix<T> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
}
///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////IMPLEMENTATIONS/////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////
impl<T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Copy + Display + Debug> Scalar for T {}

pub fn lerp <T: Tensor>(a: Tensor, b: Tensor, c: T: Scalar){}
pub fn linear_combination<T: Scalar>(
    u: impl AsRef<[Vector<T>]>,
    coefs: impl AsRef<[T]>,
) -> Vector<T> {
    u.as_ref()
        .iter()
        .zip(coefs.as_ref().iter())
        .map(|(v, c)| {
            let mut r = v.clone();
            r.scl(*c);
            r
        })
        .reduce(|mut acc, v| {
            acc.add(&v);
            acc
        })
        .unwrap()
}

impl<T: Scalar> Vector<T> {
    pub fn from(s: impl AsRef<[T]>) -> Self {
        Vector {
            data: s.as_ref().to_vec(),
        }
    }
    pub fn add(&mut self, v: &Vector<T>) {
        assert_eq!(self.data.len(), v.data.len(), "Vector size mismatch");
        self.data
            .iter_mut()
            .zip(v.data.iter())
            .for_each(|(a, b)| *a = *a + *b);
    }
    pub fn sub(&mut self, v: &Vector<T>) {
        assert_eq!(self.data.len(), v.data.len(), "Vector size mismatch");
        self.data
            .iter_mut()
            .zip(v.data.iter())
            .for_each(|(a, b)| *a = *a - *b);
    }
    pub fn scl(&mut self, s: T) {
        self.data.iter_mut().for_each(|a| *a = *a * s);
    }
}

impl<T: Scalar> Matrix<T> {
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
    fn check_dim_eq(&self, m: &Matrix<T>) {
        assert_eq!(
            self.cols, m.cols,
            "Cols mismatch {} : {}",
            self.cols, m.cols
        );
        assert_eq!(
            self.rows, m.rows,
            "Rows mismatch {} : {}",
            self.rows, m.rows
        );
    }

    pub fn add(&mut self, m: &Matrix<T>) {
        self.check_dim_eq(m);
        self.data
            .iter_mut()
            .zip(m.data.iter())
            .for_each(|(a, b)| *a = *a + *b);
    }
    pub fn sub(&mut self, m: &Matrix<T>) {
        self.check_dim_eq(m);
        self.data
            .iter_mut()
            .zip(m.data.iter())
            .for_each(|(a, b)| *a = *a - *b);
    }
    pub fn scl(&mut self, s: T) {
        self.data.iter_mut().for_each(|a| *a = *a * s);
    }
}

impl<T: Scalar> Display for Vector<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{:?}", self.data)
    }
}
impl<T: Scalar> Display for Matrix<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
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
