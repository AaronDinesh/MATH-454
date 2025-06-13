DOCNAME=main

all: report

.PHONY: clean

report:
	xelatex $(DOCNAME).tex -interation=nonstopmode
	xelatex $(DOCNAME).tex
view: report
	open $(DOCNAME).pdf

clean:
	rm -f *.blg *.bbl *.aux *.log *.lof *.lot *.out *.toc *.nav *.snm 2>/dev/null

cleanpdf:
	rm -f *.pdf 2>/dev/null
