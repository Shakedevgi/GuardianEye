"""colors.py - the palette, lifted verbatim from cv/risk_engine.py.

CLAUDE.md instruction for this phase: "Colours are already decided and
are semantically load-bearing - lift them from cv/risk_engine.py's
ZONE_COLORS and the hazard-state box colours. Note those are BGR in the cv
code." Converted to hex here, once, so nothing downstream invents its own
palette. Do not add new colours for risk/hazard meaning without checking
back against cv/risk_engine.py first - a mismatch here would mean the UI
disagrees with what's literally painted into the video pixels.
"""

# Risk zones - "how close he is" (docs/design/phase-8/Palette.dc.html)
ZONE_RED = "#FF0000"      # alert, and a clip is saved
ZONE_ORANGE = "#FF8C00"   # getting close
ZONE_YELLOW = "#FFFF00"   # in the area
ZONE_NONE = "#8C8C8C"     # the normal state

ZONE_COLOR = {
    "red": ZONE_RED,
    "orange": ZONE_ORANGE,
    "yellow": ZONE_YELLOW,
    "none": ZONE_NONE,
}

# Hazard states - CLAUDE.md decision 4's four-state table
HAZARD_CONFIRMED = "#FF0000"        # parent confirmed - alerts on approach
HAZARD_PENDING_NEW = "#FF8C00"      # arrived after first scan, unreviewed - alerts on approach
HAZARD_PENDING_FIRST_SCAN = "#00FFFF"  # found in first scan, awaiting review - no alert yet
HAZARD_DISMISSED = "#8C8C8C"        # parent said not a hazard - no alert
HAZARD_REVIEW_OUTLINE = "#FFFFFF"   # review-candidate outline, as the engine draws it
PERSON_BOX = "#00FF00"

# "Console" direction chrome (docs/design/phase-8/DirectionB_Idle.dc.html,
# DirectionB_Alert.dc.html, DirectionB_Review.dc.html) - an instrument-panel
# surface: near-black root, a persistent left rail, bordered panels. Not
# risk-semantic - see the zone/hazard colours above, which this direction
# does not change.
BG_ROOT = "#0B0E12"
BG_RAIL = "#0E1217"
BG_VIDEO_FRAME = "#0D1014"
BG_PANEL = "#0E1217"
BG_CHIP = "#161B22"
BORDER_SUBTLE = "#1B2029"
BORDER_HAIRLINE = "#1B2029"
TEXT_PRIMARY = "#E8ECF1"
TEXT_SECONDARY = "#8B949E"
# Small uppercase section labels ("ROOM", "RISK", "HAZARD MAP") - the
# mock's `.lbl` class: 11px, letter-spacing 0.1em, uppercase.
TEXT_LABEL = "#6A737D"
TEXT_MUTED = "#4F5761"
TEXT_DIM = "#5D666F"
ALERT_RED = "#FF0000"
ALERT_RED_DARK = "#1A0708"
ALERT_RED_BORDER = "#3A1114"
ALERT_TEXT_ON_RED = "#FFFFFF"
ALERT_SUBTEXT_ON_RED = "#FFE0E0"
