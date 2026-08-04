# Public-release checklist

The code package is technically organized and tested. The authors should
complete these items before pushing it to a public GitHub repository.

- [ ] Confirm with all coauthors/institutions that MIT is the intended code
  license and replace the generic copyright holder if required.
- [ ] Run the new 10-stream `(rho, beta) = (300, 48)` Section 5.3 protocol
  before public release and replace the current figure and endpoint values.
  The existing paper-facing figure was generated from the first 3 streams;
  the public frozen configuration now requires 10 streams per data set and
  80 method runs in total, matching the manuscript's stated sample count.
- [ ] Add the final repository URL, paper DOI/arXiv identifier, and publication
  metadata to `CITATION.cff` once available.
- [ ] Run `pytest` and `python scripts/validate_release.py` in a clean clone.
- [ ] Reproduce Section 5.1 from scratch and compare the generated grid counts
  and fitted slopes with the paper.
- [ ] On the target CPU cluster, reproduce the complete 15-data-set Section 5.2
  table and the 20 Section 5.3 replicate tasks from the committed configs.
- [ ] Verify that no data files, raw results, scheduler logs, PDFs, PNGs,
  credentials, or machine-specific paths are staged by Git.
