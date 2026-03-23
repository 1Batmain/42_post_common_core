use std::{
    fmt::{Debug, Display, Formatter, Result},
    ops::{Add, Deref, Mul, Sub},
};

///////////////////////////////////////////////////
///////////////////////////////////////////////////
//////////////////TYPES DECLARATION////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////
// TRAITS
pub trait Scalar:
    Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Copy + Display + Debug
{
}

pub trait TensorTrait<T: Scalar> {
    fn dim_match(&self, t: &Self) {
        assert_eq!(self.dim, t.dim)
    }
    fn add(&mut self, v: &Self);
    fn sub(&mut self, v: &Self);
    fn scl(&mut self, s: T);
}
// STRUCTS
#[derive(Clone, PartialEq, Debug)]
pub struct Tensor<T> {
    data: Vec<T>,
    dim: Vec<usize>,
}

pub struct Vector<T>(Tensor<T>);
pub struct Matrix<T>(Tensor<T>);
///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////IMPLEMENTATIONS/////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////
impl<T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Copy + Display + Debug> Scalar for T {}

impl<T> Deref for Vector<T> {
    type Target = Tensor<T>;
    fn deref(&self) -> &Tensor<T> {
        &self.0
    }
}
impl<T> Deref for Matrix<T> {
    type Target = Tensor<T>;
    fn deref(&self) -> &Tensor<T> {
        &self.0
    }
}

impl<T: Scalar> Vector<T> {
    pub fn from(s: impl AsRef<[T]>) -> Self {
        Vector(Tensor {
            data: s.as_ref().to_vec(),
            dim: vec![s.as_ref().len()],
        })
    }
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

impl<T: Scalar> TensorTrait<T> for Tensor<T> {
    fn dim_match(&self, v: &Vector<T>) {
        assert_eq!(self.data.len(), v.data.len());
    }
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
        Matrix(Tensor {
            data: s.as_ref().to_vec(),
            dim: vec![r, c],
        })
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

impl<T: Scalar> Display for Tensor<T> {
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
