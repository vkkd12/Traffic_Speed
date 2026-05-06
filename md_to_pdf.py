import pypandoc

input_file = "CHANGES.md"
output_file = "CHANGES.pdf"

pypandoc.convert_file(
    input_file,
    'pdf',
    outputfile=output_file,
    extra_args=['--pdf-engine=xelatex']
)