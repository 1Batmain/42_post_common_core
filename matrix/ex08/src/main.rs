use colored::Colorize;
use matrix::Matrix;

fn main() {
    println!(
        "{}",
        "Exercise 08 - Matrix trace".bold().underline().green()
    );

    // simple 2x2 trace
    let a = Matrix::from([1., 2., 3., 4.], 2, 2); // [[1,2],[3,4]]
    println!("a =\n{}", a);
    println!("trace(a) = {}", a.trace().unwrap().to_string().bold().red());

    // non-square matrix should error
    let b = Matrix::from([1.; 6], 3, 2); // rows=2, cols=3
    println!("\nb =\n{}", b);
    println!(
        "trace(b) -> {}",
        match b.trace() {
            Ok(v) => format!("unexpected Ok({v})").red().to_string(),
            Err(e) => format!("Err({e})").yellow().to_string(),
        }
    );
}
