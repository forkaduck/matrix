use proc_macro::TokenStream;
use quote::quote;

#[proc_macro]
pub fn matrix_new(input: TokenStream) -> TokenStream {
    if input.is_empty() {
        panic!("No input provided");
    }

    // Get a string from the TokenStream.
    let tokens = format!("{}", input).clone();
    let mut tokens = tokens.split(", ");

    // Fetch the loader reference from the input string.
    let t_loader = tokens.next().expect("Kernel Loader struct not found");

    // Retrieve the type of the matrix.
    let t_type = tokens.next().expect("Type not found");

    // Parse out the amount of dimensions.
    let t_dimensions = tokens
        .next()
        .expect("Dimension not found")
        .parse::<usize>()
        .expect("Dimension is not a number");

    let t_dimensions = t_dimensions - 1;

    // Generate the nested vectors.
    let mut inner_data = String::new();
    for _ in 0..(t_dimensions) {
        inner_data.push_str("Vec<");
    }
    inner_data.push_str(t_type);

    for _ in 0..(t_dimensions) {
        inner_data.push('>');
    }

    let inner_data: proc_macro2::TokenStream = inner_data
        .parse()
        .expect("Failed to parse nested vector statement TokenStream");

    let loader: proc_macro2::TokenStream =
        t_loader.parse().expect("Failed to parse loader reference");

    let gen = quote! {
        Matrix { data: Vec::<#inner_data>::new(), loader:#loader}
    };
    gen.into()
}
