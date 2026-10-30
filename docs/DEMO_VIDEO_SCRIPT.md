# Cross-Market Financial Quant & Filing Analyst Agent
## Capstone Demo Video Recording Script (Q3 — 40 Points)

**Target Video Duration:** 4 to 5 Minutes (Meets 3–6 min requirement)  
**Deliverable:** Screen recording with voiceover narration + working web terminal UI  
**Application URL:** `http://localhost:8000`  

---

### Video Outline & Timeline Overview

| Time | Scene / Screen | Topic / Feature Demonstrated |
| :---: | :--- | :--- |
| **0:00 – 0:45** | Terminal Web UI (`http://localhost:8000`) | Problem Statement & Architecture Overview |
| **0:45 – 1:30** | Company Snapshots & Relational DB | 100% Real Filing Universe (`BE`, `GOOG`, `JPM`, `PENG`, `SPCX`) |
| **1:30 – 2:45** | Live Chat / SSE Stream (Success Case) | Multi-Ticker Synthesis: Bloom Energy (`BE`) vs SpaceX (`SPCX`) |
| **2:45 – 3:45** | Live Chat (Honest Failure / Trap Case) | Hallucination Trap Abstention: Bitcoin guidance for ungrounded company |
| **3:45 – 4:30** | Evaluation Report & Code Walkthrough | Benchmark Results (30 Qrels, 69.45% Acc vs 25.22% Zero-Shot) |

---

### Scene-by-Scene Script & Narration

#### Scene 1: Introduction & Architecture (0:00 – 0:45)
- **On Screen:** Open browser at `http://localhost:8000/`. Show the Bloomberg-style dark mode terminal with header: *"CROSS-MARKET QUANT & FILING TERMINAL"*.
- **Narration (Spoken Voiceover):**
  > *"Hello everyone! Welcome to the demonstration of my Capstone project: the Cross-Market Financial Quant & Filing Analyst Agent.  
  > In quantitative finance and equity research, analysts face a major dilemma when using standard LLMs: language models suffer from severe hallucinations on exact financial numbers, and next-token prediction cannot reliably calculate ratios like DuPont Return on Equity or Altman Z-Scores.  
  > To solve this, I built an end-to-end, dual-engine financial analyst. It combines an autonomous ReAct agent loop, a sandboxed read-only SQLite relational database for multi-year statements, and a grounded vector retrieval pipeline built on 100% verified real SEC filings."*

---

#### Scene 2: 100% Real Filing Universe (0:45 – 1:30)
- **On Screen:** Point to the company metric cards at the top of the UI (`BE`, `GOOG`, `JPM`, `PENG`, `SPCX`). Click on the Bloom Energy card or hover over the snapshot metrics.
- **Narration:**
  > *"First, let's look at the data universe. Rather than using synthetic mock data, this system is built entirely on 14 authentic SEC Form 10-Q corporate filings for 2026.  
  > We have Bloom Energy (BE), Alphabet (GOOG), JPMorgan Chase (JPM), Penguin Solutions (PENG), and SpaceX (SPCX).  
  > In the database, we pre-structure their balance sheets and income statements into SQLite with foreign-key relations, while the textual disclosures, MD&A, and footnotes are indexed in a section-aware vector store that preserves financial tables intact."*

---

#### Scene 3: Success Case Walkthrough — Multi-Ticker Synthesis (1:30 – 2:45)
- **On Screen:** In the query box, paste the following query and hit Enter:
  ```text
  Compare Bloom Energy (BE) and SpaceX (SPCX) 2026 performance: compute their DuPont ROE, state their full-year guidance, and summarize key commercial drivers with citations.
  ```
- **Action On Screen:** Watch the live Server-Sent Events (SSE) stream. Point out the glowing tool badges popping up in real time:
  - `tool_start: query_financial_db`
  - `tool_result: query_financial_db`
  - `tool_start: calculate_quant_metrics`
  - `tool_start: search_filings_rag`
  - Streaming markdown text output
  - Verified citation tags `[BE_10-Q_2026_Q3:c1]` and `[SPCX_10-Q_2026_Q2:c0]`
- **Narration:**
  > *"Now, let's execute a complex multi-ticker comparative query. Notice what happens as I submit the request:  
  > Thanks to FastAPI and Server-Sent Events, the frontend streams the agent's internal thought process in real time.  
  > The agent first queries the sandboxed SQLite database to fetch exact revenues, net incomes, and balance sheet assets.  
  > Next, it invokes our deterministic Python quant calculator to compute the exact 3-step DuPont ROE, avoiding arithmetic hallucination.  
  > Then, it performs a section-aware vector search across the 10-Q filings to extract Bloom Energy's raised full-year guidance of $4.4B to $4.6B and SpaceX's Starlink revenue trajectory.  
  > Finally, every single quantitative claim in the synthesized report includes a verified citation tag linking back to the exact filing section."*

---

#### Scene 4: Honest Failure / Adversarial Trap Demonstration (2:45 – 3:45)
- **On Screen:** Click in the query box and enter an adversarial out-of-corpus trap query:
  ```text
  What is PTT Public Company Limited (PTT) and Bitcoin treasury allocation guidance for 2028?
  ```
- **Action On Screen:** Submit the query. Show that within 50ms, the system returns:
  - `Signal: ABSTAIN`
  - `Confidence: 0.0`
  - A polite refusal explaining that PTT is outside the active corpus, along with clear guidance on how the user can click 'Upload Filing' to add PTT.
- **Narration:**
  > *"Now, let's test the system's honesty with an adversarial trap query. Here, I'm asking about PTT Public Company and a hypothetical Bitcoin treasury allocation in 2028.  
  > A naive LLM or uncalibrated RAG pipeline would hallucinate plausible-sounding numbers.  
  > But our agent immediately activates its guardrails: it recognizes that PTT is outside the coverage universe, outputs an honest refusal with Signal ABSTAIN and Confidence 0.0, and provides helpful guidance on how to upload the missing Form 56-1 or 10-K filing through the UI. This honest abstention is what prevents multi-million dollar trading errors."*

---

#### Scene 5: Evaluation Results & Conclusion (3:45 – 4:30)
- **On Screen:** Switch to VS Code or GitHub showing `eval/results_summary.md` and the test suite terminal output (`16 passed in tests/`).
- **Narration:**
  > *"To validate the system rigorously, I built an automated evaluation harness with 30 gold-standard benchmark questions in eval/qrels.jsonl across 4 categories.  
  > Looking at the results:  
  > - Baseline 1 (Zero-Shot LLM) scored only 25.22% numerical accuracy with 0% citation precision.  
  > - Baseline 2 (Naive Fixed RAG) reached 47.41% accuracy but hallucinated on 100% of trap queries.  
  > - Our Proposed Full Agent achieved 69.45% numerical accuracy, 92.88% citation precision, and an 80% trap abstention rate.  
  > All 16 unit and end-to-end integration tests pass with 100% success.  
  > The entire project runs with Docker Compose and is fully documented in the repository. Thank you for watching!"*

---

### Tips for Recording
1. **Screen Resolution:** Record at 1920x1080 (16:9).
2. **Audio:** Use a clean microphone with minimal background noise.
3. **Speed:** Keep mouse movements smooth and deliberate when highlighting tool badges and citations.
4. **Export Format:** MP4 (H.264) with clear audio.
