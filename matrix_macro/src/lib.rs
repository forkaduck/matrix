use proc_macro::TokenStream;
use quote::quote;

/// Creates a new matrix.
///
/// * `<1>` - A reference to the KernelLoader object.
/// * `<2>` - The type that the matrix should contain.
/// * `<3>` - The dimensionality.
#[proc_macro]
pub fn matrix_new(input: TokenStream) -> TokenStream {
    // Get a string from the TokenStream.
    let tokens = format!("{}", input).clone();
    let mut tokens = tokens.split(", ");

    let syntax = [
        "matrix_new!(<loader>",
        ", <type>",
        ", <dims>",
        ", |<pre-alloc>|)",
    ];

    if input.is_empty() {
        panic!(
            "{}",
            &(syntax[0].to_owned() + " <- Kernel struct not found!")
        );
    }

    // Fetch the loader reference from the input string.
    let t_loader = tokens
        .next()
        .expect(&(syntax[0].to_owned() + " <- Kernel struct not found!"));

    // Retrieve the type of the matrix.
    let t_type = tokens
        .next()
        .expect(&(syntax[0].to_owned() + syntax[1] + " <- Type not found!"));

    // Parse out the amount of dimensions.
    let t_dimensions = tokens
        .next()
        .expect(&(syntax[0].to_owned() + syntax[1] + syntax[2] + " <- Dimension not found!"))
        .parse::<usize>()
        .expect(&(syntax[0].to_owned() + syntax[1] + syntax[2] + " <- Dimension is not a usize!"));

    let t_dimensions = t_dimensions - 1;

    let t_pre_alloc = tokens.next();

    // Generate the nested vectors.
    let mut inner_data = String::new();
    for _ in 0..(t_dimensions) {
        inner_data.push_str("Vec<");
    }
    inner_data.push_str(t_type);

    for _ in 0..(t_dimensions) {
        inner_data.push('>');
    }

    let function = match t_pre_alloc {
        Some(a) => {
            format!("with_capacity({})", a)
        }
        None => {
            format!("new()")
        }
    };

    // Parse strings to TokenStreams.
    let inner_data: proc_macro2::TokenStream = inner_data
        .parse()
        .expect("Failed to parse nested vector statement TokenStream");

    let loader: proc_macro2::TokenStream =
        t_loader.parse().expect("Failed to parse loader reference");

    let function: proc_macro2::TokenStream = function
        .parse()
        .expect("Failed to parse function TokenStream");

    let gen = quote! {
        Matrix { A: Vec::<#inner_data>::#function, loader:Some(#loader)}
    };
    gen.into()
}
