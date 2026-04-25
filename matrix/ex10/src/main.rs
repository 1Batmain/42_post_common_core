use colored::Colorize;
use matrix::Matrix;

fn main() {
    println!(
        "{}",
        "ex10 - Row Echelon Form (REF)".bold().underline().green()
    );

    #[rustfmt::skip]
    let mat = Matrix::<f32>::default();
    let mut res = mat.clone();
    println!("\n{}", "Test 1: identity matrix".bold().blue());
    println!("original:\n{}", mat);
    res.row_echelon();
    println!("REF:\n{}", res);
    #[rustfmt::skip]
    let mat = Matrix::from(vec![
        1., 0., 0., 0.,
        0., 1., 0., 0.,
        0., 0., 1., 0.,
        0., 0., 0., 1.,
    ], 4, 4);
    let mut res = mat.clone();
    println!("\n{}", "Test 1: identity matrix".bold().blue());
    println!("original:\n{}", mat);
    res.row_echelon();
    println!("REF:\n{}", res);
    #[rustfmt::skip]
    let mat = Matrix::from(vec![
        1930., 0., 0., 0.,
    ], 4, 1);
    let mut res = mat.clone();
    println!("\n{}", "Test 1: identity matrix".bold().blue());
    println!("original:\n{}", mat);
    res.row_echelon();
    println!("REF:\n{}", res);
    #[rustfmt::skip]
    let mat = Matrix::from(vec![
        1., 0., 0., 0.,
        0., 0., 0., 0.,
        0., 0., 1., 0.,
        0., 0., 0., 0.,
    ], 4, 4);
    let mut res = mat.clone();
    println!("\n{}", "Test 1: identity matrix".bold().blue());
    println!("original:\n{}", mat);
    res.row_echelon();
    println!("REF:\n{}", res);
    #[rustfmt::skip]
    let mat = Matrix::from(vec![
        0., 0., 0., 0.,
        0., 5., 0., 0.,
        0., 0., 0., 0.,
        0., 0., 0., -11.,
    ], 4, 4);
    let mut res = mat.clone();
    println!("\n{}", "Test 1: identity matrix".bold().blue());
    println!("original:\n{}", mat);
    res.row_echelon();
    println!("REF:\n{}", res);
    #[rustfmt::skip]
    let mat = Matrix::from(vec![
        78., 88., -23.,
        21., 11., 5.,
        652., 0., 13.
    ], 3, 3);
    let mut res = mat.clone();
    res.row_echelon();
    println!("\n{}", "Test 1: identity matrix".bold().blue());
    println!("original:\n{}", mat);
    println!("REF:\n{}", res);
    #[rustfmt::skip]
    let mat = Matrix::from(vec![
        1., 2., 3.,4.
    ], 2, 2);
    let mut res = mat.clone();
    res.row_echelon();
    println!("\n{}", "Test 1: 2xi2 matrix".bold().blue());
    println!("original:\n{}", mat);
    println!("REF:\n{}", res);
    #[rustfmt::skip]
    let mat = Matrix::from(vec![
        1., 2.,
        2., 4.], 2, 2);
    let mut res = mat.clone();
    res.row_echelon();
    println!("\n{}", "Test 1: 2xi2 matrix".bold().blue());
    println!("original:\n{}", mat);
    println!("REF:\n{}", res);
    #[rustfmt::skip]
    let mat = Matrix::from(vec![
        8., 5., -2., 4., 28.,
        4., 2.5, 20., 4., -4.,
        8., 5., 1., 4., 17.,
        ],
        5,3);
    let mut res = mat.clone();
    println!("\n{}", "Test 1: 2xi2 matrix".bold().blue());
    println!("original:\n{}", mat);
    res.row_echelon();
    println!("REF:\n{}", res);
}
