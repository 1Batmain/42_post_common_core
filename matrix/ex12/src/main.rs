use colored::Colorize;
use matrix::Matrix;

fn main() -> Result<(), String> {
    println!("{}", "ex12 - Matrix invert".bold().underline().green());

    #[rustfmt::skip]
    let mut mat = Matrix::from([
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0

        ],4,4,);
    println!("original matrix:\n{}", mat.inverse()?);

    Ok(())
}
