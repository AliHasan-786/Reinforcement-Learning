"""
Builds RL_HW4.pdf. Run after hw4_bayesian_rl.py.

Greek symbols are written via ReportLab Paragraph entities (e.g. &alpha;),
which the Helvetica font supports. Other strings use ASCII (alpha, beta).
"""

import os
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, HRFlowable, Preformatted,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(BASE_DIR, "RL_HW4.pdf")
IMG = {
    "fig1": os.path.join(BASE_DIR, "hw4_fig1_cumulative_regret.png"),
    "fig2": os.path.join(BASE_DIR, "hw4_fig2_posterior_distributions.png"),
    "fig3": os.path.join(BASE_DIR, "hw4_fig3_information_ratio.png"),
}

NAVY = colors.HexColor("#1B3A6B")
GRAY_ALT = colors.HexColor("#F5F7FA")


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.HexColor("#888888"))
    canvas.drawCentredString(letter[0] / 2.0, 0.45 * inch,
                             str(canvas.getPageNumber()))
    canvas.restoreState()


def build_styles():
    S = {}
    S["title"] = ParagraphStyle("Title", fontName="Helvetica-Bold",
                                fontSize=18, leading=24,
                                alignment=TA_CENTER, spaceAfter=4)
    S["subtitle"] = ParagraphStyle("Sub", fontName="Helvetica",
                                   fontSize=12, leading=16,
                                   alignment=TA_CENTER, spaceAfter=3)
    S["date"] = ParagraphStyle("Date", fontName="Helvetica",
                               fontSize=11, alignment=TA_CENTER, spaceAfter=8)
    S["h1"] = ParagraphStyle("H1", fontName="Helvetica-Bold",
                             fontSize=14, leading=18,
                             spaceAfter=4, spaceBefore=10, textColor=NAVY)
    S["h2"] = ParagraphStyle("H2", fontName="Helvetica-Bold",
                             fontSize=12, leading=15,
                             spaceAfter=3, spaceBefore=7, textColor=NAVY)
    S["body"] = ParagraphStyle("Body", fontName="Helvetica",
                               fontSize=10.5, leading=15,
                               alignment=TA_JUSTIFY, spaceAfter=5)
    S["mono"] = ParagraphStyle("Mono", fontName="Courier",
                               fontSize=9, leading=13, spaceAfter=3)
    S["caption"] = ParagraphStyle("Cap", fontName="Helvetica",
                                  fontSize=9.5, leading=13,
                                  alignment=TA_CENTER,
                                  textColor=colors.HexColor("#444444"),
                                  spaceBefore=3, spaceAfter=8)
    S["bullet"] = ParagraphStyle("Bullet", fontName="Helvetica",
                                 fontSize=10.5, leading=15,
                                 leftIndent=16, spaceAfter=3, bulletIndent=6)
    S["cell"] = ParagraphStyle("Cell", fontName="Helvetica",
                               fontSize=9, leading=12, alignment=TA_CENTER)
    S["cell_left"] = ParagraphStyle("CellLeft", fontName="Helvetica",
                                    fontSize=9, leading=12)
    S["analysis"] = ParagraphStyle("Analysis", fontName="Helvetica",
                                   fontSize=10.5, leading=16,
                                   alignment=TA_JUSTIFY, spaceAfter=6)
    S["abstract"] = ParagraphStyle("Abstract", fontName="Helvetica",
                                   fontSize=10, leading=14,
                                   alignment=TA_JUSTIFY,
                                   leftIndent=18, rightIndent=18,
                                   spaceAfter=6)
    S["abstract_h"] = ParagraphStyle("AbstractH", fontName="Helvetica-Bold",
                                     fontSize=11, leading=14,
                                     alignment=TA_CENTER, spaceAfter=4)
    return S


def cell(text, style):
    """Wrap a cell value in a Paragraph so long strings wrap inside the cell."""
    return Paragraph(text, style)


def styled_table(data, col_widths, header_rows=1):
    t = Table(data, colWidths=col_widths, repeatRows=header_rows)
    style = [
        ("BACKGROUND", (0, 0), (-1, header_rows - 1), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, header_rows - 1), colors.white),
        ("FONTNAME", (0, 0), (-1, header_rows - 1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, header_rows), (-1, -1),
         [colors.white, GRAY_ALT]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    t.setStyle(TableStyle(style))
    return t


def img_block(path, width, caption_text, styles):
    elems = []
    if os.path.exists(path):
        elems.append(Image(path, width=width, height=width * 0.625))
    else:
        elems.append(Paragraph(f"[Figure not found: {path}]", styles["body"]))
    elems.append(Paragraph(caption_text, styles["caption"]))
    return elems


def _read_source():
    """Return the source up to (but excluding) the ANALYSIS string. The
    prose inside that string is already shown in Section 5 and should not
    be re-rendered as monospace in the appendix."""
    src_path = os.path.join(BASE_DIR, "hw4_bayesian_rl.py")
    if not os.path.exists(src_path):
        return "[hw4_bayesian_rl.py not found]"
    with open(src_path) as f:
        text = f.read()
    cutoff = text.find("ANALYSIS = ")
    if cutoff != -1:
        text = text[:cutoff].rstrip() + "\n"
    return text


def _para_split(raw_text):
    paras = []
    current = []
    for line in raw_text.splitlines():
        line = line.rstrip()
        if not line and current:
            paras.append(" ".join(current))
            current = []
        elif line:
            current.append(line.strip())
    if current:
        paras.append(" ".join(current))
    return paras


def build_pdf():
    S = build_styles()
    doc = SimpleDocTemplate(
        OUTPUT_PATH, pagesize=letter,
        leftMargin=1.0 * inch, rightMargin=1.0 * inch,
        topMargin=1.0 * inch, bottomMargin=0.8 * inch,
    )
    W = letter[0] - 2.0 * inch

    story = []

    # Title block.
    story.append(Paragraph("Assignment 4: Bayesian Reinforcement Learning",
                           S["title"]))
    story.append(Paragraph(
        "Housing Track. NYC Borough Arms. K-armed bandit with Beta posteriors.",
        S["subtitle"]))
    story.append(Paragraph("Ali Hasan. Cornell University. April 30, 2026",
                           S["date"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=NAVY,
                            spaceAfter=10))

    # Abstract.
    story.append(Paragraph("Abstract", S["abstract_h"]))
    story.append(Paragraph(
        "I implement four bandit algorithms (UCB1, Bayes-UCB, Thompson "
        "Sampling, and Greedy) on a NYC borough Beta-Bernoulli setup with "
        "horizon T = 200 and 100 independent replications. The prior "
        "Beta(2, max(5, 1/cr_k)) is built from 2010-2017 historical cap "
        "rates, so the true best arm (Bronx) has the highest prior mean. "
        "Bayes-UCB achieves the lowest cumulative regret at T = 200 (mean "
        "2.99), followed by Greedy (4.26), Thompson Sampling (5.70), and "
        "UCB1 (17.13). Greedy's strong showing is a lucky alignment with "
        "the informative prior, not its O(T) worst-case bound. The Liu and "
        "Li (2015) prior-sensitivity result accounts for the gap between "
        "Bayesian methods and UCB1: a well-calibrated prior places "
        "non-trivial mass on the optimal arm, lowering the Bayesian regret "
        "bound; a flat prior would inflate it.",
        S["abstract"]))
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=colors.HexColor("#888888"),
                            spaceAfter=10))

    # 1. Problem Setup.
    story.append(Paragraph("1. Problem Setup", S["h1"]))
    story.append(Paragraph(
        "Each of the five NYC boroughs (Manhattan, Brooklyn, Queens, Bronx, "
        "Staten Island) is treated as one arm of a K-armed bandit. The latent "
        "reward parameter &theta;<sub>k</sub> is the probability that the "
        "quarterly cap-rate return for residential properties in borough k "
        "exceeds the 3% threshold. Observations are Bernoulli: Y<sub>t</sub> "
        "= 1 if the return exceeds 3%, and 0 otherwise.",
        S["body"]))

    story.append(Paragraph("Prior", S["h2"]))
    story.append(Paragraph(
        "Each arm uses Beta(&alpha;<sub>k</sub>, &beta;<sub>k</sub>) with "
        "&alpha;<sub>k</sub> = 2 and &beta;<sub>k</sub> = max(5, 1/cr_k), "
        "where cr_k is the historical 2010-2017 average cap rate for "
        "borough k. A higher historical cap rate yields a smaller "
        "&beta;<sub>k</sub>, so the prior mean is higher for boroughs with "
        "stronger historical returns.",
        S["body"]))

    prior_data = [
        ["Borough", "cr (2010-17)", "alpha_0", "beta_0",
         "Prior Mean", "True theta"],
        ["Manhattan", "3.5%", "2", "28.57", "0.065", "0.22"],
        ["Brooklyn", "4.8%", "2", "20.83", "0.088", "0.38"],
        ["Queens", "5.2%", "2", "19.23", "0.094", "0.42"],
        ["Bronx (best)", "6.5%", "2", "15.38", "0.115", "0.48"],
        ["Staten Island", "5.8%", "2", "17.24", "0.104", "0.35"],
    ]
    story.append(Spacer(1, 6))
    story.append(styled_table(
        prior_data,
        [1.35 * inch, 1.0 * inch, 0.7 * inch, 0.7 * inch, 0.9 * inch,
         0.9 * inch]))
    story.append(Paragraph(
        "Table 0. Prior parameters and synthetic ground-truth theta values. "
        "True theta values are chosen consistent with the prior ranking.",
        S["caption"]))

    story.append(Paragraph(
        "Posterior update after observing Y<sub>t</sub> from arm k: "
        "Beta(&alpha;<sub>k</sub> + Y<sub>t</sub>, &beta;<sub>k</sub> + 1 - "
        "Y<sub>t</sub>). The horizon is T = 200 quarterly decisions, "
        "covering five years.",
        S["body"]))

    # 2. Algorithms.
    story.append(Paragraph("2. Algorithms Implemented", S["h1"]))

    rows = [
        ("UCB1 (baseline)",
         "argmax_k [mu_hat_k + sqrt(2 ln t / n_k)]",
         "O(K ln T) frequentist regret", "Auer et al. 2002"),
        ("Bayes-UCB",
         "argmax_k Q(1 - 1/t, Beta(alpha_k, beta_k))",
         "Matches Lai-Robbins lower bound",
         "Kaufmann et al. 2012a, sec. 3.2"),
        ("Thompson Sampling",
         "Sample theta_k ~ Beta(alpha_k, beta_k); play argmax_k theta_k",
         "O(sqrt(KT log T)) Bayesian regret",
         "Thompson 1933, sec. 3.3"),
        ("Greedy (baseline)",
         "argmax_k posterior mean",
         "O(T) worst-case regret", "-"),
    ]
    alg_data = [["Algorithm", "Action Rule", "Theoretical Guarantee",
                 "Reference"]]
    for r in rows:
        alg_data.append([cell(c, S["cell"]) for c in r])
    story.append(styled_table(
        alg_data,
        [1.2 * inch, 2.0 * inch, 1.85 * inch, 1.2 * inch]))
    story.append(Paragraph("Table 1. Algorithm summary.", S["caption"]))

    # 3. Implementation.
    story.append(Paragraph("3. Implementation Notes", S["h1"]))
    bullets = [
        "<b>BetaPosterior class:</b> methods "
        "<font face='Courier'>update(y)</font>, "
        "<font face='Courier'>sample()</font>, "
        "<font face='Courier'>mean()</font>, "
        "<font face='Courier'>quantile(q)</font>. The quantile method calls "
        "<font face='Courier'>scipy.stats.beta.ppf</font>.",

        "<b>Regret tracking:</b> cumulative pseudo-regret R(t) = "
        "sum over i=1..t of (theta* - theta_{a_i}), logged at every step.",

        "<b>Confidence rule:</b> Bayes-UCB uses the (1 - 1/t) posterior "
        "quantile; UCB1 uses sqrt(2 ln t / n_k).",

        "<b>Replications:</b> each algorithm is run 100 times with distinct "
        "seeds. Reported as mean and 1.96 SE.",

        "<b>Information ratio numerator:</b> per Eq. 3.1 of Ghavamzadeh et "
        "al. (2015), the numerator is (E[Delta_t])^2. We estimate E[Delta_t] "
        "by averaging the instantaneous regret across the 100 Thompson "
        "Sampling replications at each t, then square. A 95% bootstrap CI is "
        "computed by resampling replications.",
    ]
    for b in bullets:
        story.append(Paragraph(f"&bull; {b}", S["bullet"]))
        story.append(Spacer(1, 2))

    story.append(PageBreak())

    # Figure 1.
    story.append(Paragraph("4. Results", S["h1"]))
    story.append(Paragraph("Figure 1. Cumulative regret vs. t", S["h2"]))
    story += img_block(
        IMG["fig1"], W,
        "Figure 1. Cumulative pseudo-regret R(t) for all four algorithms over "
        "T = 200 steps; shaded bands are 95% CIs across 100 replications. "
        "Bayes-UCB achieves the lowest final regret, followed by Greedy and "
        "Thompson Sampling. UCB1, which has no prior, accumulates the most "
        "regret because it pays a full exploration cost. Greedy performs well "
        "in this run because the informative prior already ranks Bronx (the "
        "true best arm) first by prior mean, so greedy locks onto the best "
        "arm at step 1 and never leaves; this is the lucky regime, not the "
        "guaranteed one (see analysis).",
        S)

    story.append(PageBreak())

    # Figure 2.
    story.append(Paragraph("Figure 2. Posterior evolution", S["h2"]))
    story += img_block(
        IMG["fig2"], W,
        "Figure 2. Posterior distributions Beta(alpha_k, beta_k) for each "
        "borough at t = 50, 100, and 200, from one representative Thompson "
        "Sampling run. Dotted vertical lines mark the true theta_k; the "
        "dashed black line marks theta* = 0.48 (Bronx). By t = 200 the Bronx "
        "posterior is concentrated near its true value; the other arms "
        "remain near their priors because TS rarely pulls them.",
        S)

    story.append(PageBreak())

    # Figure 3.
    story.append(Paragraph("Figure 3. Information-ratio numerator", S["h2"]))
    story += img_block(
        IMG["fig3"], W,
        "Figure 3. Per-step information-ratio numerator (E[Delta_t])^2 for "
        "Thompson Sampling, where Delta_t = theta* - theta_{a_t} is the "
        "instantaneous regret. The expectation is estimated by averaging "
        "Delta_t across the 100 replications at each t and squaring. Top: "
        "rolling mean (window = 15) with a 95% bootstrap CI. Bottom: "
        "running average. The numerator decays as TS concentrates posterior "
        "mass on Bronx and pulls a sub-optimal arm less often.",
        S)

    story.append(PageBreak())

    # Table 2.
    story.append(Paragraph("Table 2. Final cumulative regret at T = 200",
                           S["h2"]))

    reg_path = os.path.join(BASE_DIR, "hw4_regret_table.csv")
    if os.path.exists(reg_path):
        df = pd.read_csv(reg_path)
        reg_data = [["Algorithm", "Mean R(T)", "SE", "95% CI",
                     "Theoretical Guarantee"]]
        for _, row in df.iterrows():
            reg_data.append([
                str(row["Algorithm"]),
                f"{row['Mean']:.3f}",
                f"{row['SE']:.3f}",
                f"[{row['CI_lo']:.3f}, {row['CI_hi']:.3f}]",
                str(row["Guarantee"]),
            ])
    else:
        reg_data = [["Algorithm", "Mean", "SE", "95% CI", "Guarantee"],
                    ["-", "-", "-", "-", "Run hw4_bayesian_rl.py first"]]

    story.append(styled_table(
        reg_data,
        [1.4 * inch, 0.85 * inch, 0.65 * inch, 1.35 * inch, 2.0 * inch]))
    story.append(Paragraph(
        "Table 2. Final cumulative pseudo-regret at T = 200, "
        "mean and 1.96 SE across 100 replications.",
        S["caption"]))

    story.append(Spacer(1, 8))

    # Analysis.
    story.append(Paragraph("5. Analysis", S["h1"]))
    analysis_path = os.path.join(BASE_DIR, "hw4_analysis.txt")
    if os.path.exists(analysis_path):
        with open(analysis_path) as f:
            raw = f.read()
        for pt in _para_split(raw):
            story.append(Paragraph(pt, S["analysis"]))
            story.append(Spacer(1, 4))
    else:
        story.append(Paragraph(
            "(Analysis not found; run hw4_bayesian_rl.py first.)",
            S["analysis"]))

    # 6. Conclusion.
    story.append(Paragraph("6. Conclusion", S["h1"]))
    story.append(Paragraph(
        "The borough bandit gave a clean test bed for the four algorithms. "
        "Bayes-UCB came out clearly ahead, which lines up with its "
        "Lai-Robbins-matching guarantee. Thompson Sampling was close behind. "
        "The surprise was Greedy: it ran second on regret here, but only "
        "because the historical-cap-rate prior happened to rank Bronx first. "
        "Under a flat or flipped prior its O(T) bound would bite, exactly "
        "the regime Liu and Li (2015) describe with their O(sqrt(T/p)) "
        "bad-prior bound. UCB1 paid the expected upfront exploration cost "
        "without a prior to lean on. The information-ratio numerator decays "
        "smoothly as TS concentrates posterior mass on Bronx, which is what "
        "the Russo-Van Roy framework predicts.",
        S["body"]))

    # 7. References.
    story.append(Paragraph("References", S["h1"]))
    refs = [
        "[1] Auer, P., Cesa-Bianchi, N., &amp; Fischer, P. (2002). "
        "Finite-time analysis of the multiarmed bandit problem. "
        "Machine Learning, 47(2-3), 235-256.",

        "[2] Kaufmann, E., Capp&eacute;, O., &amp; Garivier, A. (2012). "
        "On Bayesian upper confidence bounds for bandit problems. "
        "Proceedings of AISTATS, 592-600.",

        "[3] Thompson, W.R. (1933). On the likelihood that one unknown "
        "probability exceeds another in view of the evidence of two "
        "samples. Biometrika, 25(3-4), 285-294.",

        "[4] Liu, C.-Y., &amp; Li, L. (2015). On the prior sensitivity of "
        "Thompson Sampling. arXiv:1506.03378.",

        "[5] Ghavamzadeh, M., Mannor, S., Pineau, J., &amp; Tamar, A. "
        "(2015). Bayesian reinforcement learning: A survey. Foundations "
        "and Trends in Machine Learning, 8(5-6), 359-483.",

        "[6] Russo, D., &amp; Van Roy, B. (2016). An information-theoretic "
        "analysis of Thompson sampling. Journal of Machine Learning "
        "Research, 17(68), 1-30.",
    ]
    ref_style = ParagraphStyle(
        "Ref", fontName="Helvetica", fontSize=10, leading=14,
        leftIndent=18, firstLineIndent=-18, spaceAfter=4,
    )
    for r in refs:
        story.append(Paragraph(r, ref_style))

    story.append(PageBreak())

    # Code appendix. Preformatted preserves whitespace/indentation; we feed it
    # ~45-line chunks so each chunk can break across pages cleanly.
    story.append(Paragraph("Appendix. Source Code (hw4_bayesian_rl.py)",
                           S["h1"]))
    code_text = _read_source()
    code_lines = code_text.splitlines()
    chunk_size = 45
    for i in range(0, len(code_lines), chunk_size):
        block = "\n".join(code_lines[i:i + chunk_size])
        story.append(Preformatted(block, S["mono"]))

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    print(f"Report written to {OUTPUT_PATH}")


if __name__ == "__main__":
    build_pdf()
