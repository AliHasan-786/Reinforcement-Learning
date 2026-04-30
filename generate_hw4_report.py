"""
generate_hw4_report.py
Generates RL_HW4.pdf — Assignment 4: Bayesian RL (Housing Track)
Run AFTER hw4_bayesian_rl.py has generated all figures.
"""

import os
import inspect
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, HRFlowable, KeepTogether
)

# ─── paths ──────────────────────────────────────────────────
BASE_DIR    = "/Users/alihasan/Downloads/Reinforcement Learning"
OUTPUT_PATH = os.path.join(BASE_DIR, "RL_HW4.pdf")
IMG         = {
    "fig1": os.path.join(BASE_DIR, "hw4_fig1_cumulative_regret.png"),
    "fig2": os.path.join(BASE_DIR, "hw4_fig2_posterior_distributions.png"),
    "fig3": os.path.join(BASE_DIR, "hw4_fig3_information_ratio.png"),
}

# ─── colours ────────────────────────────────────────────────
NAVY       = colors.HexColor("#1B3A6B")
RED        = colors.HexColor("#B31B1B")
GRAY_HDR   = colors.HexColor("#E8ECF0")
GRAY_ALT   = colors.HexColor("#F5F7FA")
RULE_COLOR = colors.HexColor("#2C3E50")

# ─── footer ─────────────────────────────────────────────────
def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.HexColor("#888888"))
    canvas.drawCentredString(letter[0] / 2.0, 0.45 * inch, str(canvas.getPageNumber()))
    canvas.restoreState()


# ─── styles ─────────────────────────────────────────────────
def build_styles():
    S = {}
    S["title"] = ParagraphStyle("Title", fontName="Helvetica-Bold",
                                fontSize=18, leading=24, alignment=TA_CENTER, spaceAfter=4)
    S["subtitle"] = ParagraphStyle("Sub", fontName="Helvetica",
                                   fontSize=12, leading=16, alignment=TA_CENTER, spaceAfter=3)
    S["date"]   = ParagraphStyle("Date", fontName="Helvetica",
                                 fontSize=11, alignment=TA_CENTER, spaceAfter=8)
    S["h1"]     = ParagraphStyle("H1", fontName="Helvetica-Bold",
                                 fontSize=14, leading=18, spaceAfter=4, spaceBefore=10,
                                 textColor=NAVY)
    S["h2"]     = ParagraphStyle("H2", fontName="Helvetica-Bold",
                                 fontSize=12, leading=15, spaceAfter=3, spaceBefore=7,
                                 textColor=NAVY)
    S["body"]   = ParagraphStyle("Body", fontName="Helvetica",
                                 fontSize=10.5, leading=15, alignment=TA_JUSTIFY,
                                 spaceAfter=5)
    S["mono"]   = ParagraphStyle("Mono", fontName="Courier",
                                 fontSize=9, leading=13, spaceAfter=3)
    S["caption"]= ParagraphStyle("Cap", fontName="Helvetica",
                                 fontSize=9.5, leading=13, alignment=TA_CENTER,
                                 textColor=colors.HexColor("#444444"), spaceBefore=3, spaceAfter=8)
    S["bullet"] = ParagraphStyle("Bullet", fontName="Helvetica",
                                 fontSize=10.5, leading=15, leftIndent=16, spaceAfter=3,
                                 bulletIndent=6)
    S["analysis"] = ParagraphStyle("Analysis", fontName="Helvetica",
                                   fontSize=10.5, leading=16, alignment=TA_JUSTIFY,
                                   spaceAfter=6)
    return S


# ─── table helper ───────────────────────────────────────────
def styled_table(data, col_widths, header_rows=1):
    t = Table(data, colWidths=col_widths, repeatRows=header_rows)
    style = [
        ("BACKGROUND",  (0, 0), (-1, header_rows - 1), NAVY),
        ("TEXTCOLOR",   (0, 0), (-1, header_rows - 1), colors.white),
        ("FONTNAME",    (0, 0), (-1, header_rows - 1), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9.5),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, header_rows), (-1, -1), [colors.white, GRAY_ALT]),
        ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0),(-1, -1), 5),
    ]
    t.setStyle(TableStyle(style))
    return t


# ─── image helper ───────────────────────────────────────────
def img_block(path, width, caption_text, styles):
    elems = []
    if os.path.exists(path):
        elems.append(Image(path, width=width, height=width * 0.625))
    else:
        elems.append(Paragraph(f"[Figure not found: {path}]", styles["body"]))
    elems.append(Paragraph(caption_text, styles["caption"]))
    return elems


# ─── HW4 code excerpt (auto-read) ───────────────────────────
def _read_source(max_lines=160):
    src_path = os.path.join(BASE_DIR, "hw4_bayesian_rl.py")
    if not os.path.exists(src_path):
        return "[hw4_bayesian_rl.py not found]"
    with open(src_path) as f:
        lines = f.readlines()
    return "".join(lines[:max_lines])


# ============================================================
# BUILD DOCUMENT
# ============================================================

def build_pdf():
    S = build_styles()
    doc = SimpleDocTemplate(
        OUTPUT_PATH,
        pagesize=letter,
        leftMargin=1.0 * inch, rightMargin=1.0 * inch,
        topMargin=1.0 * inch,  bottomMargin=0.8 * inch,
    )
    W = letter[0] - 2.0 * inch   # usable text width

    story = []

    # ── Title block ─────────────────────────────────────────
    story.append(Paragraph("Assignment 4 — Bayesian Reinforcement Learning", S["title"]))
    story.append(Paragraph("Housing Track: NYC Borough Arms  ·  K-Armed Bandit with Beta Posteriors", S["subtitle"]))
    story.append(Paragraph("Ali Hasan  ·  Cornell University  ·  April 30, 2026", S["date"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=NAVY, spaceAfter=10))

    # ── 1. Problem Setup ────────────────────────────────────
    story.append(Paragraph("1. Problem Setup", S["h1"]))
    story.append(Paragraph(
        "Each of the five NYC boroughs — Manhattan, Brooklyn, Queens, Bronx, and Staten Island — "
        "is treated as an arm of a K-armed bandit. The latent reward parameter θ<sub>k</sub> "
        "represents the probability that the quarterly cap-rate return for residential properties "
        "in borough k exceeds the 3% threshold. Observations are Bernoulli: Y<sub>t</sub> = 1 "
        "if the return exceeds 3%, and 0 otherwise.",
        S["body"]))

    story.append(Paragraph("Prior Specification", S["h2"]))
    story.append(Paragraph(
        "Using historical NYC residential cap rates (2010–2017), each arm's prior is "
        "Beta(α<sub>k</sub>, β<sub>k</sub>) with α<sub>k</sub> = 2 and "
        "β<sub>k</sub> = max(5, 1/ĉr<sub>k</sub>). A higher historical cap rate yields a "
        "smaller β (i.e., a less conservative prior), consistent with greater prior confidence "
        "that returns will exceed the threshold.",
        S["body"]))

    prior_data = [
        ["Borough", "ĉr (2010–17)", "α₀", "β₀", "Prior Mean", "True θ"],
        ["Manhattan",    "3.5%", "2", "28.57", "0.065", "0.22"],
        ["Brooklyn",     "4.8%", "2", "20.83", "0.088", "0.38"],
        ["Queens",       "5.2%", "2", "19.23", "0.094", "0.42"],
        ["Bronx ★",      "6.5%", "2", "15.38", "0.115", "0.48"],
        ["Staten Island","5.8%", "2", "17.24", "0.104", "0.35"],
    ]
    story.append(Spacer(1, 6))
    story.append(styled_table(prior_data,
                              [1.35*inch, 1.0*inch, 0.5*inch, 0.7*inch, 0.9*inch, 0.8*inch]))
    story.append(Paragraph(
        "Table 0 — Prior parameters and true θ values. ★ denotes the optimal arm.",
        S["caption"]))

    story.append(Paragraph(
        "Posterior update after observing Y<sub>t</sub> from arm k: "
        "Beta(α<sub>k</sub> + Y<sub>t</sub>,  β<sub>k</sub> + 1 − Y<sub>t</sub>). "
        "The horizon is T = 200 quarterly investment decisions.",
        S["body"]))

    # ── 2. Algorithms ───────────────────────────────────────
    story.append(Paragraph("2. Algorithms Implemented", S["h1"]))

    alg_data = [
        ["Algorithm",     "Action Rule",                          "Theoretical Guarantee",           "Reference"],
        ["UCB1\n(baseline)", "argmax_k [μ̂_k + √(2 ln t / n_k)]", "O(K ln T) frequentist regret",  "Auer et al. 2002"],
        ["Bayes-UCB",     "argmax_k Q(1−1/t, Beta(α_k, β_k))",   "Matches Lai-Robbins lower bound","Kaufmann et al. 2012a, §3.2"],
        ["Thompson\nSampling","Sample θ̂_k~Beta(α_k,β_k); play argmax_k θ̂_k",
                                                                   "O(√(KT log T)) Bayesian regret","Thompson 1933, §3.3"],
        ["Greedy\n(baseline)","argmax_k posterior mean",          "O(T) worst-case regret",          "—"],
    ]
    story.append(styled_table(alg_data,
                              [1.1*inch, 2.1*inch, 1.85*inch, 1.2*inch]))
    story.append(Paragraph("Table 1 — Algorithm summary.", S["caption"]))

    # ── 3. Implementation ────────────────────────────────────
    story.append(Paragraph("3. Implementation Notes", S["h1"]))
    bullets = [
        "<b>BetaPosterior class:</b> methods <font face='Courier'>update(y)</font>, "
        "<font face='Courier'>sample()</font>, <font face='Courier'>mean()</font>, "
        "<font face='Courier'>quantile(q)</font> (uses <font face='Courier'>scipy.stats.beta.ppf</font>).",

        "<b>Regret tracking:</b> Cumulative pseudo-regret "
        "R(t) = Σ<sub>i=1</sub><sup>t</sup> (θ* − θ<sub>a_i</sub>), computed at every step.",

        "<b>Confidence bands:</b> Bayes-UCB uses the (1 − 1/t) posterior quantile; "
        "UCB1 uses the frequentist √(2 ln t / n<sub>k</sub>) bonus.",

        "<b>Replications:</b> Each algorithm is run 100 times with distinct random seeds. "
        "Reported as mean ± 1.96 × SE.",

        "<b>Information ratio:</b> Γ<sub>t</sub> numerator = (Δ<sub>t</sub>)² = "
        "(θ* − θ<sub>a_t</sub>)², the squared instantaneous pseudo-regret, as defined "
        "in Eq. 3.1 of Ghavamzadeh et al. (2015).",
    ]
    for b in bullets:
        story.append(Paragraph(f"• {b}", S["bullet"]))
        story.append(Spacer(1, 2))

    story.append(PageBreak())

    # ── Figure 1 ────────────────────────────────────────────
    story.append(Paragraph("4. Results", S["h1"]))
    story.append(Paragraph("Figure 1 — Cumulative Regret vs. t", S["h2"]))
    story += img_block(IMG["fig1"], W,
        "Figure 1. Cumulative pseudo-regret R(t) for all four algorithms over T = 200 steps. "
        "Shaded bands = 95 % confidence intervals across 100 replications. "
        "Thompson Sampling and Bayes-UCB accumulate the lowest regret, confirming the "
        "theoretical advantage of Bayesian exploration under an informative prior. "
        "UCB1 shows moderate regret from pure frequentist exploration, "
        "while the Greedy baseline diverges due to insufficient exploration.", S)

    story.append(PageBreak())

    # ── Figure 2 ────────────────────────────────────────────
    story.append(Paragraph("Figure 2 — Posterior Evolution", S["h2"]))
    story += img_block(IMG["fig2"], W,
        "Figure 2. Posterior distributions Beta(α_k, β_k) for each borough arm at "
        "t = 50, 100, and 200 (single representative replication). "
        "Dotted vertical lines = true θ_k; dashed black = optimal arm θ* = 0.48 (Bronx). "
        "By t = 200, the Bronx posterior is tightly concentrated near the true value, "
        "while other arms have been largely eliminated.", S)

    story.append(PageBreak())

    # ── Figure 3 ────────────────────────────────────────────
    story.append(Paragraph("Figure 3 — Information Ratio Numerator", S["h2"]))
    story += img_block(IMG["fig3"], W,
        "Figure 3. Rolling mean (window = 15) of the information-ratio numerator "
        "Γ_t = (Δ_t)² = (θ* − θ_{a_t})² for Thompson Sampling across 100 replications. "
        "Top: per-step trajectory with 95 % CI (faint traces = individual replications). "
        "Bottom: cumulative running average, showing rapid decay as TS identifies the "
        "Bronx arm and incurs near-zero per-step regret.", S)

    story.append(PageBreak())

    # ── Table 2 — Final Regret ───────────────────────────────
    story.append(Paragraph("Table 2 — Final Cumulative Regret at T = 200", S["h2"]))

    reg_path = os.path.join(BASE_DIR, "hw4_regret_table.csv")
    if os.path.exists(reg_path):
        import pandas as pd
        df = pd.read_csv(reg_path)
        reg_data = [["Algorithm", "Mean R(T)", "SE", "95 % CI", "Theoretical Guarantee"]]
        for _, row in df.iterrows():
            reg_data.append([
                str(row["Algorithm"]),
                f"{row['Mean']:.3f}",
                f"{row['SE']:.3f}",
                f"[{row['CI_lo']:.3f}, {row['CI_hi']:.3f}]",
                str(row["Guarantee"]),
            ])
    else:
        reg_data = [["Algorithm","Mean","SE","95% CI","Guarantee"],
                    ["—","—","—","—","Run hw4_bayesian_rl.py first"]]

    story.append(styled_table(reg_data,
                              [1.4*inch, 0.85*inch, 0.65*inch, 1.35*inch, 2.0*inch]))
    story.append(Paragraph(
        "Table 2 — Final cumulative pseudo-regret statistics at T = 200. "
        "Mean ± 1.96 SE across 100 independent replications.", S["caption"]))

    story.append(Spacer(1, 8))

    # ── Analysis ────────────────────────────────────────────
    story.append(Paragraph("5. Analysis", S["h1"]))

    analysis_path = os.path.join(BASE_DIR, "hw4_analysis.txt")
    if os.path.exists(analysis_path):
        with open(analysis_path) as f:
            raw = f.read()
        # strip header lines from the txt
        lines = [l for l in raw.splitlines() if not l.startswith("ANALYSIS") and not l.startswith("─")]
        body = " ".join(l.strip() for l in lines if l.strip())
    else:
        body = "(Analysis not found — run hw4_bayesian_rl.py first.)"

    # Split into readable paragraphs at blank lines
    para_texts = []
    current = []
    for line in (analysis_path and open(analysis_path).readlines() or []):
        line = line.rstrip()
        if not line and current:
            para_texts.append(" ".join(current))
            current = []
        elif line and not line.startswith("ANALYSIS") and not line.startswith("─"):
            current.append(line.strip())
    if current:
        para_texts.append(" ".join(current))

    for pt in para_texts:
        if pt.strip():
            story.append(Paragraph(pt, S["analysis"]))
            story.append(Spacer(1, 4))

    story.append(PageBreak())

    # ── Code Appendix ────────────────────────────────────────
    story.append(Paragraph("Appendix — Source Code (hw4_bayesian_rl.py)", S["h1"]))
    code_text = _read_source(max_lines=200)
    for chunk in code_text.split("\n"):
        # escape XML characters
        chunk = (chunk
                 .replace("&", "&amp;")
                 .replace("<", "&lt;")
                 .replace(">", "&gt;"))
        story.append(Paragraph(chunk if chunk.strip() else "&nbsp;", S["mono"]))

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    print(f"\n  Report saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    build_pdf()
