use colored::Colorize;
use matrix::Matrix;

fn main() {
    println!(
        "{}",
        "ex10 - Row Echelon Form (REF)".bold().underline().green()
    );

    #[rustfmt::skip]
    let mat = Matrix::<f32>::default();
    println!("mat:\n{}", mat);
    println!(
        "Δ = {}",
        match mat.determinant() {
            Ok(value) => value.to_string(),
            Err(e) => e,
        }
    );
    #[rustfmt::skip]
    let mat = Matrix::from(vec![
        1., 0., 0., 0.,
        0., 1., 0., 0.,
        0., 0., 1., 0.,
        0., 0., 0., 1.,
    ], 4, 4);
    println!("mat:\n{}", mat);
    println!(
        "Δ = {}",
        match mat.determinant() {
            Ok(value) => value.to_string(),
            Err(e) => e,
        }
    );
    #[rustfmt::skip]
    let mat = Matrix::from(vec![
        1930., 0., 0., 0.,
    ], 4, 1);
    println!("mat:\n{}", mat);
    println!(
        "Δ = {}",
        match mat.determinant() {
            Ok(value) => value.to_string(),
            Err(e) => e,
        }
    );
    #[rustfmt::skip]
    let mat = Matrix::from(vec![
        1., 0., 0., 0.,
        0., 0., 0., 0.,
        0., 0., 1., 0.,
        0., 0., 0., 0.,
    ], 4, 4);
    println!("mat:\n{}", mat);
    println!(
        "Δ = {}",
        match mat.determinant() {
            Ok(value) => value.to_string(),
            Err(e) => e,
        }
    );
    #[rustfmt::skip]
    let mat = Matrix::from(vec![
        0., 0., 0., 0.,
        0., 5., 0., 0.,
        0., 0., 0., 0.,
        0., 0., 0., -11.,
    ], 4, 4);
    println!("mat:\n{}", mat);
    println!(
        "Δ = {}",
        match mat.determinant() {
            Ok(value) => value.to_string(),
            Err(e) => e,
        }
    );
    #[rustfmt::skip]
    let mat = Matrix::from(vec![
        78., 88., -23.,
        21., 11., 5.,
        652., 0., 13.
    ], 3, 3);
    println!("mat:\n{}", mat);
    println!(
        "Δ = {}",
        match mat.determinant() {
            Ok(value) => value.to_string(),
            Err(e) => e,
        }
    );
    #[rustfmt::skip]
    let mat = Matrix::from(vec![
        1., 2., 3.,4.
    ], 2, 2);
    println!("mat:\n{}", mat);
    println!(
        "Δ = {}",
        match mat.determinant() {
            Ok(value) => value.to_string(),
            Err(e) => e,
        }
    );
    #[rustfmt::skip]
    let mat = Matrix::from(vec![
        1., 2.,
        2., 4.], 2, 2);
    println!("mat:\n{}", mat);
    println!(
        "Δ = {}",
        match mat.determinant() {
            Ok(value) => value.to_string(),
            Err(e) => e,
        }
    );
    #[rustfmt::skip]
    let mat = Matrix::from(vec![
        8., 5., -2., 4., 28.,
        4., 2.5, 20., 4., -4.,
        8., 5., 1., 4., 17.,
        ],
        5,3);
    println!("mat:\n{}", mat);
    println!(
        "Δ = {}",
        match mat.determinant() {
            Ok(value) => value.to_string(),
            Err(e) => e,
        }
    );
}
