# Docling Migration Evaluation for RAGDoll

## Executive Summary

**Docling** is IBM's open-source document understanding toolkit that could potentially replace RAGDoll's current multi-tool extraction pipeline with a unified solution. This document evaluates the fit and outlines migration considerations.

---

## Current RAGDoll Architecture

### Extraction Stack
- **PDF**: PyMuPDF (fitz) for text + pdfplumber for tables + custom heuristics for charts/figures
- **DOCX**: python-docx for paragraphs/tables + custom image extraction
- **Excel**: openpyxl/pandas for sheets → table regions
- **Images**: pytesseract OCR + custom classification (text/table/chart/figure)
- **Custom Logic**: Heuristic-based detection of charts vs figures vs tables

### Processing Pipeline
1. Extract structured regions (text blocks, charts, tables, figures)
2. OCR images for charts/figures
3. LLM interpretation of non-prose content (charts, tables, figures)
4. Embed only summaries (not raw artifacts)
5. Store raw artifacts separately

---

## Docling Capabilities

### What Docling Offers

**Unified Document Parsing:**
- Single library for PDF, DOCX, PPTX, XLSX, HTML, images
- AI-powered layout analysis (DocLayNet model)
- Table structure recognition (TableFormer)
- Code and formula detection
- Image classification
- Reading order detection

**Output Formats:**
- Markdown (structured)
- HTML
- DocTags (structured JSON)
- JSON (full document structure)

**Performance:**
- ~30x faster than OCR-based approaches (avoids OCR when possible)
- Trained on 81,000 manually labeled pages (patents, manuals, 10-K filings)

**Integration:**
- Works with LangChain, LlamaIndex, spaCy, Crew AI, Haystack
- Python API (5 lines of code to set up)
- CLI tool available
- Local execution (good for sensitive data)

---

## Fit Analysis

### ✅ Advantages of Migrating to Docling

1. **Better Table Extraction**
   - Current: pdfplumber can miss complex tables, struggles with merged cells
   - Docling: TableFormer model specifically trained for table structure
   - **Impact**: Higher quality table data → better LLM interpretations

2. **Improved Layout Understanding**
   - Current: Heuristic-based detection (char count, drawing detection)
   - Docling: AI model trained on diverse document types
   - **Impact**: More accurate chart/figure detection, better reading order

3. **Unified Extraction**
   - Current: Multiple libraries (PyMuPDF, pdfplumber, python-docx, openpyxl)
   - Docling: Single library for most formats
   - **Impact**: Simpler codebase, fewer dependencies, easier maintenance

4. **Better OCR Integration**
   - Current: Manual pytesseract calls
   - Docling: Built-in OCR with smart fallback (only when needed)
   - **Impact**: Faster processing, fewer OCR errors

5. **Structured Output**
   - Current: Custom dataclasses (`Document`, `TextBlock`, `ChartRegion`, etc.)
   - Docling: Standardized DoclingDocument format
   - **Impact**: More consistent data structure, easier to extend

6. **Performance**
   - Current: OCR all images, even when text is extractable
   - Docling: Avoids OCR when text is available (~30x faster claim)
   - **Impact**: Faster ingestion, especially for large document sets

### ⚠️ Challenges & Considerations

1. **LLM Interpretation Workflow**
   - **Current**: RAGDoll uses LLM to interpret charts/tables/figures (qualitative summaries)
   - **Docling**: Provides structured data but doesn't do interpretation
   - **Decision**: Keep LLM interpretation step? Or rely on Docling's structured output?
   - **Recommendation**: Keep LLM interpretation for semantic summaries, but use Docling's better structure detection

2. **Custom Artifact Storage**
   - **Current**: Stores raw images/JSON separately in `artifacts/` directory
   - **Docling**: Outputs structured JSON but doesn't handle storage
   - **Decision**: Continue storing artifacts separately or embed in Docling output?
   - **Recommendation**: Keep separate storage for backward compatibility

3. **Figure vs Chart Detection**
   - **Current**: Custom heuristics (drawings + char count, short blocks)
   - **Docling**: Has image classification but may not distinguish charts from figures
   - **Decision**: Use Docling's classification or keep custom logic?
   - **Recommendation**: Evaluate Docling's classification quality first

4. **Dependency Changes**
   - **Current**: PyMuPDF, pdfplumber, python-docx, openpyxl, pytesseract
   - **Docling**: New dependency (requires Python ≥3.10)
   - **Impact**: Need to update requirements, test compatibility
   - **Note**: Docling may still use some underlying libraries

5. **Migration Effort**
   - **Scope**: Replace `extractors.py` (330+ lines), update `watcher.py` integration
   - **Testing**: Need to reprocess existing documents to compare quality
   - **Risk**: Breaking changes to document structure detection

6. **Backward Compatibility**
   - **Current**: Existing DB/JSONL uses current structure
   - **Migration**: May need to reprocess documents or handle both formats
   - **Recommendation**: Make it configurable (use Docling or legacy)

---

## Migration Plan (If Proceeding)

### Phase 1: Proof of Concept (1-2 weeks)
1. Install Docling, test on sample documents
2. Compare extraction quality vs current system
3. Evaluate table/chart/figure detection accuracy
4. Benchmark performance (speed, memory)

### Phase 2: Integration Design (1 week)
1. Design adapter layer: DoclingDocument → RAGDoll Document
2. Plan LLM interpretation integration
3. Design artifact storage strategy
4. Plan backward compatibility approach

### Phase 3: Implementation (2-3 weeks)
1. Create new `extractors_docling.py` module
2. Implement adapter: convert Docling output to RAGDoll format
3. Update `watcher.py` to use new extractor (with feature flag)
4. Keep legacy extractor as fallback

### Phase 4: Testing & Validation (1-2 weeks)
1. Reprocess sample documents with both systems
2. Compare chunk quality, embedding similarity
3. Test edge cases (complex layouts, scanned PDFs, etc.)
4. Performance benchmarking

### Phase 5: Rollout (1 week)
1. Make Docling the default (with legacy fallback)
2. Monitor production usage
3. Collect feedback on quality improvements
4. Eventually deprecate legacy extractor

---

## Recommendation

### **Hybrid Approach (Recommended)**

1. **Adopt Docling for table extraction** (highest value)
   - Replace pdfplumber with Docling's TableFormer
   - Keep current text extraction (PyMuPDF works well)
   - Keep current chart/figure detection (or enhance with Docling)

2. **Gradual Migration**
   - Add Docling as optional dependency
   - Feature flag: `RAGDOLL_USE_DOCLING=true`
   - Run both systems in parallel, compare results
   - Switch to Docling when confident

3. **Keep LLM Interpretation**
   - Docling provides structure, but LLM provides semantic meaning
   - Continue using LLM for chart/table/figure summaries
   - Use Docling's better structure to improve LLM prompts

### **Full Migration (If Quality is Significantly Better)**

If testing shows Docling provides significantly better extraction:
- Replace entire extraction pipeline
- Use Docling's structured output
- Keep LLM interpretation step
- Maintain artifact storage approach

---

## Key Questions to Answer

Before committing to migration, test:

1. **Table Quality**: Does Docling extract tables better than pdfplumber?
   - Test on complex tables, merged cells, multi-page tables

2. **Chart/Figure Detection**: Does Docling's image classification match your needs?
   - Test on your document types
   - Compare with current heuristics

3. **Performance**: Is Docling actually faster on your documents?
   - Benchmark on representative document set
   - Measure memory usage

4. **Output Format**: Can you easily convert Docling output to RAGDoll format?
   - Test adapter implementation
   - Verify all needed data is available

5. **Dependencies**: Are there conflicts with current stack?
   - Check Python version requirements
   - Verify no breaking changes to other tools

---

## Next Steps

1. **Quick Test** (1-2 hours):
   ```bash
   pip install docling
   # Test on a few sample PDFs
   # Compare table extraction quality
   ```

2. **Evaluate Results**:
   - Is table extraction noticeably better?
   - Does layout detection work for your document types?
   - Is performance acceptable?

3. **Decision Point**:
   - If quality is significantly better → proceed with migration
   - If marginal improvement → consider hybrid approach
   - If no improvement → stick with current system

---

## Resources

- **Docling Documentation**: https://docling-project.github.io/docling/
- **PyPI**: https://pypi.org/project/docling/
- **Research Paper**: https://research.ibm.com/publications/docling-an-efficient-open-source-toolkit-for-ai-driven-document-conversion
- **IBM Tutorial**: https://www.ibm.com/think/tutorials/build-document-question-answering-system-with-docling-and-granite

---

## Conclusion

Docling offers promising improvements, especially for table extraction and layout understanding. However, RAGDoll's current system works and has custom logic tailored to your workflow (LLM interpretation, artifact storage).

**Recommendation**: Start with a proof-of-concept test. If Docling shows significant quality improvements (especially for tables), proceed with a hybrid or gradual migration. If improvements are marginal, the migration effort may not be worth it.

The key is to **test on your actual documents** before committing to a full migration.
