"""Render a QBRData object as an editable PowerPoint (native charts, real tables, speaker notes)."""

from __future__ import annotations

import io
import math

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import (XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_MARKER_STYLE,
                             XL_TICK_LABEL_POSITION)
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

from cca.qbr import QBRData

NAVY = RGBColor(0x1B, 0x2A, 0x41)      # dominant: title/closing backgrounds, headings
INK = RGBColor(0x1F, 0x29, 0x37)       # body text
MUTED = RGBColor(0x6B, 0x72, 0x80)     # captions
CARD = RGBColor(0xF2, 0xF4, 0xF7)      # card fill
OVER = RGBColor(0xC2, 0x41, 0x0C)      # over budget / cost increase
UNDER = RGBColor(0x0F, 0x76, 0x6E)     # under budget / cost decrease
ACCENT = RGBColor(0x2A, 0x78, 0xD6)    # forecast / neutral series
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
HEAD, BODY = "Cambria", "Calibri"
W, H = Inches(13.333), Inches(7.5)
M = Inches(0.6)


def _nice_step(raw: float) -> float:
    """Round an axis step to 1, 2, 2.5 or 5 x 10^k so ticks land on clean values."""
    mag = 10 ** math.floor(math.log10(raw))
    return next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)


def _money(v: float, signed: bool = False) -> str:
    a = abs(v)
    s = f"{a / 1e6:,.2f}M" if a >= 1e6 else f"{a / 1e3:,.1f}K" if a >= 1e3 else f"{a:,.0f}"
    sign = ("+" if v >= 0 else "-") if signed else ("-" if v < 0 else "")
    return f"{sign}${s}"


def _text(slide, x, y, w, h, text, size=14, bold=False, color=INK, font=BODY, align=PP_ALIGN.LEFT,
          anchor=MSO_ANCHOR.TOP, italic=False):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    lines = text if isinstance(text, list) else [text]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.size, r.font.bold, r.font.italic = Pt(size), bold, italic
        r.font.color.rgb, r.font.name = color, font
    return tb


def _bullets(slide, x, y, w, h, items, size=15, color=INK):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(9)
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(Emu(Inches(0.25))))
        pPr.set("indent", str(-Emu(Inches(0.25))))
        bu = pPr.makeelement("{http://schemas.openxmlformats.org/drawingml/2006/main}buChar", {"char": "•"})
        pPr.append(bu)
        r = p.add_run()
        r.text = item
        r.font.size, r.font.color.rgb, r.font.name = Pt(size), color, BODY
    return tb


def _title(slide, text, subtitle=None):
    ph = slide.shapes.title
    ph.left, ph.top, ph.width, ph.height = M, Inches(0.45), W - 2 * M, Inches(0.8)
    tf = ph.text_frame
    tf.margin_left = tf.margin_right = 0
    tf.vertical_anchor = MSO_ANCHOR.TOP
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    r = p.add_run()
    r.text = text
    # one line at 32pt holds ~58 characters on this slide width; shrink rather than wrap
    size = 32 if len(text) <= 58 else 28 if len(text) <= 66 else 24
    r.font.size, r.font.bold, r.font.name, r.font.color.rgb = Pt(size), True, HEAD, NAVY
    if subtitle:
        _text(slide, M, Inches(1.2), W - 2 * M, Inches(0.4), subtitle, size=14, color=MUTED)


def _card(slide, x, y, w, h, label, value, sub=None, value_color=NAVY):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    box.adjustments[0] = 0.08
    box.fill.solid()
    box.fill.fore_color.rgb = CARD
    box.line.fill.background()
    box.shadow.inherit = False
    pad = Inches(0.25)
    _text(slide, x + pad, y + Inches(0.2), w - 2 * pad, Inches(0.35), label, size=12, color=MUTED)
    _text(slide, x + pad, y + Inches(0.55), w - 2 * pad, Inches(0.75), value, size=34, bold=True,
          color=value_color, font=HEAD)
    if sub:
        _text(slide, x + pad, y + h - Inches(0.55), w - 2 * pad, Inches(0.4), sub, size=12, color=MUTED)


def _footer(slide, n, dark=False):
    _text(slide, M, H - Inches(0.45), Inches(9), Inches(0.3),
          "Cloud Cost Analyst · Microsoft Azure Public Dataset V2 (CC BY 4.0) · list-price costs",
          size=10, color=WHITE if dark else MUTED)
    _text(slide, W - M - Inches(1), H - Inches(0.45), Inches(1), Inches(0.3), str(n), size=10,
          color=WHITE if dark else MUTED, align=PP_ALIGN.RIGHT)


def _bg(slide, color):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def _style_chart(chart, legend=False):
    chart.font.size, chart.font.name = Pt(12), BODY
    chart.font.color.rgb = INK
    chart.has_legend = legend
    if legend:
        chart.legend.position = XL_LEGEND_POSITION.BOTTOM
        chart.legend.include_in_layout = False
        chart.legend.font.size = Pt(12)


def _table(slide, x, y, w, rows, col_w, header_fill=NAVY, size=12, row_h=Inches(0.42), aligns=None):
    shape = slide.shapes.add_table(len(rows), len(rows[0]), x, y, w, row_h * len(rows))
    tbl = shape.table
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = cw
    for i, row in enumerate(rows):
        tbl.rows[i].height = row_h
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            color = None
            if isinstance(val, tuple):
                val, color = val
            cell.text = str(val)
            cell.margin_left = cell.margin_right = Inches(0.1)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.alignment = (aligns[j] if aligns else PP_ALIGN.LEFT)
            f = p.runs[0].font
            f.size, f.name = Pt(size), BODY
            cell.fill.solid()
            if i == 0:
                cell.fill.fore_color.rgb = header_fill
                f.bold, f.color.rgb = True, WHITE
            else:
                cell.fill.fore_color.rgb = WHITE if i % 2 else CARD
                f.color.rgb = color or INK
    return shape


def build_pptx(d: QBRData) -> bytes:
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    title_only, blank = prs.slide_layouts[5], prs.slide_layouts[6]
    ps, pe = d.period
    t = d.total
    over = t["variance_usd"] >= 0
    vcol = OVER if over else UNDER
    n = 0

    # 1 -- title (dark)
    s = prs.slides.add_slide(blank)
    _bg(s, NAVY)
    n += 1
    _text(s, M, Inches(2.3), Inches(11), Inches(1.2), "Cloud Cost Review", size=48, bold=True, color=WHITE, font=HEAD)
    _text(s, M, Inches(3.5), Inches(11), Inches(0.6), f"Days {ps}–{pe}: budget vs. actual, root causes, savings and outlook",
          size=20, color=WHITE)
    _text(s, M, Inches(4.4), Inches(11), Inches(0.9),
          ["Generated automatically by the Cloud Cost Analyst agent.",
           f"Every figure in the commentary was checked by the Auditor: {d.audit.numbers_checked} numbers matched, "
           f"{d.audit.facts_reconciled} facts re-run against source."], size=14, color=RGBColor(0xC9, 0xD3, 0xE0))
    _footer(s, n, dark=True)
    s.notes_slide.notes_text_frame.text = "Purpose: review cloud spend for the period against plan, explain the variances, and agree next steps."

    # 2 -- executive summary
    s = prs.slides.add_slide(title_only)
    n += 1
    _title(s, f"Spend ran {abs(t['variance_pct']):.1%} {'over' if over else 'under'} plan in days {ps}–{pe}")
    cw, gap, y = Inches(2.85), Inches(0.27), Inches(1.45)
    _card(s, M, y, cw, Inches(1.75), "Actual spend", _money(t["actual_usd"]), f"Budget {_money(t['budget_usd'])}")
    _card(s, M + (cw + gap), y, cw, Inches(1.75), "Variance vs budget", _money(t["variance_usd"], signed=True),
          f"{t['variance_pct']:+.1%} · {'over' if over else 'under'} plan", value_color=vcol)
    _card(s, M + 2 * (cw + gap), y, cw, Inches(1.75), "Hard savings available", _money(d.savings["hard_savings_usd"]),
          "per month · idle + rightsizing", value_color=UNDER)
    _card(s, M + 3 * (cw + gap), y, cw, Inches(1.75), "Next 7 days (forecast)", _money(d.outlook["total"]),
          f"90% range {_money(d.outlook['total_lower'])}–{_money(d.outlook['total_upper'])}", value_color=ACCENT)
    _text(s, M, Inches(3.5), Inches(6), Inches(0.35), "Audited commentary", size=16, bold=True, color=NAVY, font=HEAD)
    _bullets(s, M, Inches(3.95), W - 2 * M, Inches(2.8), d.commentary, size=15)
    _footer(s, n)
    s.notes_slide.notes_text_frame.text = ("All numbers in the commentary are drawn from the fact ledger and verified by the "
                                           "deterministic Auditor, which re-runs every fact from the warehouse.")

    # 3 -- by workload category
    s = prs.slides.add_slide(title_only)
    n += 1
    cats = d.by_category.sort_values("actual_usd", ascending=False)
    top_cat = d.by_category.iloc[0]
    _title(s, f"{top_cat['name']} workloads drove the largest category variance",
           f"Run-rate budget set on day {d.plan[1]} from days {d.plan[0]}–{d.plan[1]}; actuals for days {ps}–{pe}")
    cw = Inches(3.85)
    for i, r in enumerate(cats.itertuples()):
        x = M + i * (cw + Inches(0.29))
        c = OVER if r.variance_usd >= 0 else UNDER
        _card(s, x, Inches(1.9), cw, Inches(2.2), f"{r.name} · actual {_money(r.actual_usd)}",
              f"{r.variance_pct:+.1%}", f"{_money(r.variance_usd, signed=True)} vs budget {_money(r.budget_usd)}",
              value_color=c)
    cd = CategoryChartData()
    cd.categories = list(cats.name)
    cd.add_series("Budget", [round(v) for v in cats.budget_usd])
    cd.add_series("Actual", [round(v) for v in cats.actual_usd])
    gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, M, Inches(4.35), W - 2 * M, Inches(2.6), cd)
    ch = gf.chart
    _style_chart(ch, legend=True)
    ch.plots[0].gap_width = 60
    ch.series[0].format.fill.solid()
    ch.series[0].format.fill.fore_color.rgb = RGBColor(0xB8, 0xC2, 0xD0)
    ch.series[1].format.fill.solid()
    ch.series[1].format.fill.fore_color.rgb = NAVY
    ch.value_axis.tick_labels.number_format = '$#,##0.0,,"M"'
    ch.value_axis.tick_labels.number_format_is_linked = False
    ch.value_axis.major_gridlines.format.line.color.rgb = RGBColor(0xE5, 0xE7, 0xEB)
    ch.category_axis.reverse_order = True
    _footer(s, n)
    s.notes_slide.notes_text_frame.text = "Category view: where the variance sits by Microsoft's workload label."

    # 4 -- variance by team (diverging bar)
    s = prs.slides.add_slide(title_only)
    n += 1
    teams = d.by_team.head(8)
    rest = float(d.by_team.iloc[8:].variance_usd.sum())
    names = list(teams.name) + ["All other teams"]
    vals = [float(v) for v in teams.variance_usd] + [rest]
    n_rest = len(d.by_team) - 8
    if abs(rest) >= max(abs(v) for v in vals[:-1]):
        headline = f"Team swings offset; {n_rest:,} smaller teams added {_money(rest, signed=True)}"
    else:
        headline = "A handful of teams explain most of the variance"
    _title(s, headline, f"Variance vs budget, days {ps}–{pe} (positive = over budget)")
    cd = CategoryChartData()
    cd.categories = names
    cd.add_series("Variance vs budget", [round(v) for v in vals])
    gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, M, Inches(1.75), Inches(8.2), Inches(5.0), cd)
    ch = gf.chart
    _style_chart(ch)
    ch.has_title = False
    lo, hi = min(min(vals), 0) * 1.45, max(vals) * 1.25        # room for labels outside both bar ends
    unit = _nice_step((hi - lo) / 5)
    ch.value_axis.minimum_scale = math.floor(lo / unit) * unit
    ch.value_axis.maximum_scale = math.ceil(hi / unit) * unit
    ch.value_axis.major_unit = unit
    plot = ch.plots[0]
    plot.gap_width = 45
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format, dl.number_format_is_linked = '$#,##0.0,"K";-$#,##0.0,"K"', False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size, dl.font.color.rgb = Pt(11), INK
    for i, v in enumerate(vals):
        pt = ch.series[0].points[i]
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = OVER if v >= 0 else UNDER
    ch.series[0].invert_if_negative = False
    ch.category_axis.reverse_order = True
    ch.category_axis.tick_label_position = XL_TICK_LABEL_POSITION.LOW  # labels stay left of negative bars
    ch.value_axis.tick_labels.number_format = '$#,##0,"K";-$#,##0,"K"'
    ch.value_axis.tick_labels.number_format_is_linked = False
    ch.value_axis.major_gridlines.format.line.color.rgb = RGBColor(0xE5, 0xE7, 0xEB)
    big = d.team_drivers[0]
    _text(s, Inches(9.2), Inches(1.95), Inches(3.5), Inches(0.35), "Biggest mover", size=13, color=MUTED)
    _text(s, Inches(9.2), Inches(2.3), Inches(3.5), Inches(0.7), big["team"], size=30, bold=True, color=NAVY, font=HEAD)
    _text(s, Inches(9.2), Inches(3.05), Inches(3.5), Inches(1.9),
          [f"{_money(big['variance_usd'], signed=True)} vs budget ({big['variance_pct']:+.0%})",
           f"Driven by {big['top_deployment']} ({big['what_happened']}),",
           f"{big['deployment_share']:.0%} of the team's total movement."], size=15)
    _text(s, Inches(9.2), Inches(5.0), Inches(3.5), Inches(1.4),
          f"Unbudgeted spend: {_money(t['unbudgeted_usd'])} from teams with no spend in the plan week.",
          size=13, color=MUTED, italic=True)
    _footer(s, n)
    s.notes_slide.notes_text_frame.text = "Orange bars are over budget, teal under. 'All other teams' nets everyone outside the top 8."

    # 5 -- root cause table
    s = prs.slides.add_slide(title_only)
    n += 1
    _title(s, "Each variance traces to a specific deployment",
           f"Root cause: change in each team's daily spend, plan week (days {d.plan[0]}–{d.plan[1]}) vs review period")
    rows = [["Team", "Budget", "Actual", "Variance", "Top deployment", "What happened", "Share of movement"]]
    for r in d.team_drivers:
        c = OVER if r["variance_usd"] >= 0 else UNDER
        what = r["what_happened"] + (f" (day {r['deployment_first_day']})" if r["what_happened"] == "launched" else "")
        rows.append([r["team"], _money(r["budget_usd"]), _money(r["actual_usd"]),
                     (_money(r["variance_usd"], signed=True), c), r["top_deployment"], what,
                     f"{r['deployment_share']:.0%}" if r["deployment_share"] is not None else "-"])
    R, L = PP_ALIGN.RIGHT, PP_ALIGN.LEFT
    _table(s, M, Inches(1.8), W - 2 * M, rows,
           [Inches(1.5), Inches(1.6), Inches(1.6), Inches(1.7), Inches(1.9), Inches(2.0), Inches(1.833)],
           aligns=[L, R, R, R, L, L, R], size=13)
    _text(s, M, Inches(6.05), W - 2 * M, Inches(0.6),
          "Top deployment = the largest mover in the same direction as the variance. Share of movement = its share of "
          "the team's total change in daily spend across all deployments. Decomposition is exact to the cent.",
          size=12, color=MUTED, italic=True)
    _footer(s, n)
    s.notes_slide.notes_text_frame.text = "Use this table to route each variance to the owning team for confirmation."

    # 6 -- anomalies
    s = prs.slides.add_slide(title_only)
    n += 1
    _title(s, f"{len(d.anomalies)} material anomalies in the period, each with a named cause",
           "Robust seasonal detector on the 50 largest teams; materiality floor ≈ $2K/month")
    rows = [["Team", "Started", "Type", "Excess vs expected", "Deviation", "Cause (deployment)"]]
    for a in d.anomalies:
        c = OVER if a["excess_cost_usd"] >= 0 else UNDER
        rows.append([a["series"], f"day {a['start_day']}, {a['start_hour'] % 24:02d}:00",
                     f"{'temporary' if a['pattern'] == 'transient' else 'persistent'} {a['direction']}",
                     (_money(a["excess_cost_usd"], signed=True), c), (f"{a['pct_vs_expected']:+.0%}", c),
                     f"{a['top_deployment']} ({a['what_happened']})"])
    _table(s, M, Inches(1.8), W - 2 * M, rows,
           [Inches(1.6), Inches(1.9), Inches(2.2), Inches(2.0), Inches(1.5), Inches(2.933)],
           aligns=[L, L, L, R, R, L], size=14, row_h=Inches(0.55))
    _footer(s, n)
    persistent = sum(1 for a in d.anomalies if a["pattern"] == "sustained")
    _text(s, M, Inches(1.8) + Inches(0.55) * (len(rows)) + Inches(0.35), W - 2 * M, Inches(0.8),
          f"{persistent} of {len(d.anomalies)} persisted after the alert, so they are new run-rate rather than one-off "
          "blips: confirm with each owner that the change is planned and update the budget.", size=15, color=INK)
    s.notes_slide.notes_text_frame.text = "Persistent = spend stayed at the new level after the alert; temporary = it came back."

    # 7 -- savings
    s = prs.slides.add_slide(title_only)
    n += 1
    lev = sorted([lv for lv in d.savings["levers"] if lv["lever"] in ("Rightsize over-provisioned VMs", "Shut down idle VMs")],
                 key=lambda lv: -lv["savings_usd"])
    hard = d.savings["hard_savings_usd"]
    _title(s, f"{_money(hard)} a month of savings needs no new discounts",
           "Idle shutdown and rightsizing, priced at list rates; commitment and Spot discounts are contract-dependent")
    _card(s, M, Inches(1.9), Inches(3.6), Inches(2.1), "Hard savings per month", _money(hard),
          f"{hard / d.savings['total_cost_usd']:.1%} of monthly spend", value_color=UNDER)
    vms = sum(int(lv["vms"] or 0) for lv in lev)
    _card(s, M, Inches(4.25), Inches(3.6), Inches(2.1), "VMs to act on", f"{vms:,}", "idle or over-provisioned")
    cd = CategoryChartData()
    cd.categories = [lv["lever"] for lv in lev]
    cd.add_series("Savings per month", [round(lv["savings_usd"]) for lv in lev])
    gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(4.6), Inches(1.9), Inches(8.1), Inches(2.6), cd)
    ch = gf.chart
    _style_chart(ch)
    ch.has_title = False
    ch.plots[0].gap_width = 50
    ch.plots[0].has_data_labels = True
    ch.plots[0].data_labels.number_format = '$#,##0.00,,"M"'
    ch.plots[0].data_labels.number_format_is_linked = False
    ch.plots[0].data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
    ch.series[0].format.fill.solid()
    ch.series[0].format.fill.fore_color.rgb = UNDER
    ch.value_axis.visible = False
    ch.value_axis.has_major_gridlines = False
    ch.category_axis.reverse_order = True
    _bullets(s, Inches(4.6), Inches(4.75), Inches(8.1), Inches(1.8), [
        "Rightsizing: VMs whose P95 CPU stays under 40% move one size down (Azure v5 prices halve per step).",
        "Idle: VMs under 1% average CPU for 3+ days; shut down or schedule off after owner review.",
        "Then commit the remaining always-on baseline: optimize first, commit second.",
    ], size=14)
    _footer(s, n)
    s.notes_slide.notes_text_frame.text = "Savings are 30-day run-rate. Spot and commitment savings depend on contract rates and overlap with rightsizing."

    # 8 -- outlook
    s = prs.slides.add_slide(title_only)
    n += 1
    o = d.outlook
    _title(s, f"Next 7 days: {_money(o['total'])} forecast, in line with recent run-rate",
           "Daily spend: actuals for days 17–30, forecast for days 31–37 with 90% interval")
    cd = CategoryChartData()
    cd.categories = [str(x) for x in o["history_days"] + o["forecast_days"]]
    blank_h, blank_f = [None] * len(o["history"]), [None] * len(o["forecast"])
    cd.add_series("Actual", [round(v) for v in o["history"]] + blank_f)
    cd.add_series("Forecast", blank_h[:-1] + [round(o["history"][-1])] + [round(v) for v in o["forecast"]])
    cd.add_series("90% interval (upper)", blank_h + [round(v) for v in o["upper"]])
    cd.add_series("90% interval (lower)", blank_h + [round(v) for v in o["lower"]])
    gf = s.shapes.add_chart(XL_CHART_TYPE.LINE, M, Inches(1.75), Inches(8.6), Inches(5.0), cd)
    ch = gf.chart
    _style_chart(ch, legend=True)
    styles = [(INK, 2.25, None), (ACCENT, 2.25, None), (ACCENT, 1.25, MSO_LINE_DASH_STYLE.DASH),
              (ACCENT, 1.25, MSO_LINE_DASH_STYLE.DASH)]
    for ser, (col, wpt, dash) in zip(ch.series, styles):
        ser.smooth = False
        ln = ser.format.line
        ln.color.rgb, ln.width = col, Pt(wpt)
        if dash:
            ln.dash_style = dash
        ser.marker.style = XL_MARKER_STYLE.NONE
    va = ch.value_axis
    lo = min(min(o["history"]), min(o["lower"]))
    hi = max(max(o["history"]), max(o["upper"]))
    va.minimum_scale, va.maximum_scale = round(lo * 0.97, -4), round(hi * 1.02, -4)
    va.tick_labels.number_format, va.tick_labels.number_format_is_linked = '$#,##0,"K"', False
    va.major_gridlines.format.line.color.rgb = RGBColor(0xE5, 0xE7, 0xEB)
    ch.category_axis.tick_labels.font.size = Pt(11)
    ch.category_axis.has_title = True
    ch.category_axis.axis_title.text_frame.text = "Trace day"
    ch.category_axis.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(11)
    ch.category_axis.axis_title.text_frame.paragraphs[0].runs[0].font.color.rgb = MUTED
    _card(s, Inches(9.6), Inches(1.95), Inches(3.13), Inches(2.0), "Forecast, next 7 days", _money(o["total"]),
          f"{_money(o['total_lower'])} – {_money(o['total_upper'])} (90%)", value_color=ACCENT)
    _text(s, Inches(9.6), Inches(4.25), Inches(3.13), Inches(2.4),
          ["Model: self-selecting blend of two from-scratch forecasters.",
           "Back-tested next-week error on total spend: 0.91% (Holt-Winters 2.06%).",
           "Intervals are learned from the model's own out-of-sample errors."], size=13, color=MUTED)
    _footer(s, n)
    s.notes_slide.notes_text_frame.text = "Note: the y-axis does not start at zero, to make day-to-day movement visible."

    # 9 -- method and audit (dark)
    s = prs.slides.add_slide(blank)
    _bg(s, NAVY)
    n += 1
    _text(s, M, Inches(0.6), Inches(11), Inches(0.8), "Method and audit trail", size=34, bold=True, color=WHITE, font=HEAD)
    recon_pass = int((d.reconciliation.status == "PASS").sum())
    _bullets(s, M, Inches(1.7), Inches(12), Inches(4.6), [
        "Data: Microsoft Azure Public Dataset V2 — 30 days of Microsoft's own first-party VM usage "
        "(2.7M VMs, 6,687 subscriptions, 33,205 deployments), priced at Azure Linux pay-as-you-go list prices.",
        f"Budget: run-rate method — average daily spend of days {d.plan[0]}–{d.plan[1]} × days in the review period.",
        f"Data quality: {recon_pass} of {len(d.reconciliation)} reconciliation checks pass; every total ties back "
        "to the raw VM table.",
        f"Commentary audit: {d.audit.numbers_checked} numbers matched cited facts and {d.audit.facts_reconciled} facts "
        "were re-run from the warehouse by the deterministic Auditor before this deck was generated.",
        "Root cause: team changes are decomposed exactly by deployment (launched, shut down, scaled up/down); "
        "'possible migration' flags are heuristics for owners to confirm.",
        "Limits: 2019 usage at current list prices (not Microsoft's actual cost); Spot and commitment discounts are inputs.",
    ], size=15, color=WHITE)
    _footer(s, n, dark=True)
    s.notes_slide.notes_text_frame.text = "Reproduce everything from the GitHub repository: github.com/bshrey0601/Cloud-Cost-Analysis"

    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()


if __name__ == "__main__":
    import argparse

    from cca.qbr import build_qbr
    from cca.warehouse import active_db_path, connect

    ap = argparse.ArgumentParser(description="Generate the QBR PowerPoint pack")
    ap.add_argument("--out", default="cloud_cost_review.pptx")
    args = ap.parse_args()
    con = connect()
    data = build_qbr(con, str(active_db_path()))
    with open(args.out, "wb") as fh:
        fh.write(build_pptx(data))
    print(f"wrote {args.out} -- commentary audit: {data.audit.summary()}")
