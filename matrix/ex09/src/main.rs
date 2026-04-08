use colored::Colorize;
use matrix::Matrix;

fn main() {
    println!("{}", "ex09 - transpose".bold().underline().green());

    // Test 1: 2x4 -> 4x2 transpose
    let mat = Matrix::from(vec![1., 2., 3., 4., 5., 6., 7., 8.], 4, 2);
    let transposed = mat.transpose();
    let expected = Matrix::from(vec![1., 5., 2., 6., 3., 7., 4., 8.], 2, 4);

    println!("\n{}", "Test 1: rectangular matrix".bold().blue());
    println!("original:\n{}", mat);
    println!("transpose:\n{}", transposed);
    println!("expected:\n{}", expected);
    println!(
        "result: {}",
        if transposed == expected {
            "OK".bold().green()
        } else {
            "KO".bold().red()
        }
    );

    // Test 2: 1x1 transpose
    let mat = Matrix::from(vec![1.], 1, 1);
    let transposed = mat.transpose();

    println!("\n{}", "Test 2: 1x1 matrix".bold().blue());
    println!("original:\n{}", mat);
    println!("transpose:\n{}", transposed);
    println!(
        "result: {}",
        if transposed == mat {
            "OK".bold().green()
        } else {
            "KO".bold().red()
        }
    );
}
