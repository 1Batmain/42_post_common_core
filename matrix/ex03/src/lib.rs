use std::{
    fmt::{Debug, Display, Formatter, Result},
    iter::Sum,
    ops::{Add, Mul, Sub},
};

///////////////////////////////////////////////////
///////////////////////////////////////////////////
//////////////////TYPES DECLARATION////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////

pub trait Scalar:
    Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Copy + Display + Debug + Sum
{
}
impl<T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Copy + Display + Debug + Sum> Scalar
    for T
{
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

pub trait Tensor<T> {
    fn add(&mut self, op: &Self);
    fn sub(&mut self, op: &Self);
    fn scl(&mut self, op: T);
}

///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////IMPLEMENTATIONS/////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////

pub fn lerp<T: Scalar, S: Tensor<T>>(a: &S, b: &S, mix: T) -> S
where
    S: Clone,
{
    let mut res = a.clone();
    let mut diff = b.clone();
    diff.sub(a);
    diff.scl(mix);
    res.add(&diff);
    res
}

pub fn dot<T: Scalar>(a: &Vector<T>, b: &Vector<T>) -> T {
    let a = &a.data;
    let b = &b.data;
    assert_eq!(a.len(), b.len(), "The two vectors are not of the same size");
    a.iter().zip(b.iter()).map(|(a, b)| *a * *b).sum()
}

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
}
impl<T: Scalar> Tensor<T> for Vector<T> {
    fn add(&mut self, v: &Vector<T>) {
        assert_eq!(self.data.len(), v.data.len(), "Vector size mismatch");
        self.data
            .iter_mut()
            .zip(v.data.iter())
            .for_each(|(a, b)| *a = *a + *b);
    }
    fn sub(&mut self, v: &Vector<T>) {
        assert_eq!(self.data.len(), v.data.len(), "Vector size mismatch");
        self.data
            .iter_mut()
            .zip(v.data.iter())
            .for_each(|(a, b)| *a = *a - *b);
    }
    fn scl(&mut self, s: T) {
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
}
impl<T: Scalar> Tensor<T> for Matrix<T> {
    fn add(&mut self, m: &Matrix<T>) {
        self.check_dim_eq(m);
        self.data
            .iter_mut()
            .zip(m.data.iter())
            .for_each(|(a, b)| *a = *a + *b);
    }
    fn sub(&mut self, m: &Matrix<T>) {
        self.check_dim_eq(m);
        self.data
            .iter_mut()
            .zip(m.data.iter())
            .for_each(|(a, b)| *a = *a - *b);
    }
    fn scl(&mut self, s: T) {
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
