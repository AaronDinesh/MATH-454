DOCNAME=main

all: report

.PHONY: clean

report:
	pdflatex $(DOCNAME).tex -interation=nonstopmode
	bibtex $(DOCNAME).aux
	pdflatex $(DOCNAME).tex
	pdflatex $(DOCNAME).tex
	pdflatex $(DOCNAME).tex
view: report
	open $(DOCNAME).pdf

clean:
	rm *.blg *.bbl *.aux *.log *.lof *.lot *.out *.toc

cleanpdf:
	rm *.pdf
