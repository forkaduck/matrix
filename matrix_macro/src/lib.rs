use proc_macro::TokenStream;
use quote::quote;

#[proc_macro]
pub fn matrix_new(input: TokenStream) -> TokenStream {
    if input.is_empty() {
        panic!("No input provided");
    }

    // Get a string from the TokenStream.
    let input = format!("{}", input).clone();
    let mut input = input.split("; ");

    // Retrieve the type of the matrix.
    let m_type = input.next().expect("Type not found");

    // Parse out the amount of dimensions.
    let m_dimensions = input
        .next()
        .expect("Dimension not found")
        .parse::<usize>()
        .expect("Dimension is not a number");

    if m_dimensions < 1 {
        panic!("Matricies with 0 dimensions are not allowed");
    }

    // Generate the nested vectors.
    let mut inner_data = String::new();
    for _ in 0..(m_dimensions - 1) {
        inner_data.push_str("Vec<");
    }
    inner_data.push_str(m_type);

    for _ in 0..(m_dimensions - 1) {
        inner_data.push('>');
    }

    let inner_data: proc_macro2::TokenStream = inner_data
        .parse()
        .expect("Failed to parse generated statement to TokenStream");

    let gen = quote! {
        Matrix { data: Vec::<#inner_data>::new()}
    };
    gen.into()
}
