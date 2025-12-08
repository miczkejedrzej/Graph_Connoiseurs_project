#!/bin/bash

pandoc report/report.md \
  --pdf-engine=xelatex \
  -V geometry:margin=1in \
  --number-sections \
  --toc \
  --highlight-style=tango \
  -H report/style.tex \
  -o report/report.pdf
