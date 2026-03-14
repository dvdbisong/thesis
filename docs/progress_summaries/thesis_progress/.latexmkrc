$pdf_mode = 5;  # Use xelatex
$postscript_mode = $dvi_mode = 0;

# Override pdflatex to use xelatex (for editors that force pdflatex)
$pdflatex = 'xelatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';
