use colored::Colorize;
use ex02::{Matrix, Vector, lerp};

fn main() {
    ///////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////
    //////////////////////VECTORS TESTS////////////////////////////
    ///////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////
    println!("{}", "Testing lerp with vectors".bold().underline().green());
    let a = Vector::from([1.; 2]);
    let b = Vector::from([-1.; 2]);
    let mut mix = 0.5;
    let res = lerp(&a, &b, mix);
    println!(
        "{}",
        format!(
            "a->{}\nb->{}\nlerp({})->{}",
            a,
            b,
            mix.to_string().green(),
            res.to_string().red().bold()
        )
    );
    mix = 0.;
    let res = lerp(&a, &b, mix);
    println!(
        "{}",
        format!(
            "a->{}\nb->{}\nlerp({})->{}",
            a,
            b,
            mix.to_string().green(),
            res.to_string().red().bold()
        )
    );
    mix = 1.;
    let res = lerp(&a, &b, mix);
    println!(
        "{}",
        format!(
            "a->{}\nb->{}\nlerp({})->{}\n",
            a,
            b,
            mix.to_string().green(),
            res.to_string().red().bold()
        )
    );
    ///////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////
    //////////////////////VECTORS TESTS////////////////////////////
    ///////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////
    println!("{}", "Testing lerp with matrixs".bold().underline().green());
    let c = Matrix::from([1.; 9], 3, 3);
    let d = Matrix::from([-1.; 9], 3, 3);
    let mut mix = 0.5;
    let res = lerp(&c, &d, mix);
    println!(
        "{}",
        format!(
            "c->{}\nd->{}\nlerp({})->{}",
            c,
            d,
            mix.to_string().green(),
            res.to_string().red().bold()
        )
    );
    mix = 0.;
    let res = lerp(&c, &d, mix);
    println!(
        "{}",
        format!(
            "c->{}\nd->{}\nlerp({})->{}",
            c,
            d,
            mix.to_string().green(),
            res.to_string().red().bold()
        )
    );
    mix = 1.;
    let res = lerp(&c, &d, mix);
    println!(
        "{}",
        format!(
            "c->{}\nd->{}\nlerp({})->{}\n",
            c,
            d,
            mix.to_string().green(),
            res.to_string().red().bold()
        )
    );
    //Doesnt compile -> matrix lerp vector
    // let res = lerp(&a, &d, mix);
    // let res = lerp(&d, &a, mix);

    // panics! not same shape
    println!("{}", "Should panic".bold().underline().red());
    let a2 = Vector::from([3.; 6]);
    let res = lerp(&a, &a2, mix);
    println!(
        "{}",
        format!(
            "a->{}\nb->{}\nlerp({})->{}",
            a,
            a2,
            mix.to_string().green(),
            res.to_string().red().bold()
        )
    );
}
