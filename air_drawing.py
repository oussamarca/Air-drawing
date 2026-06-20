import cv2
import mediapipe as mp
import numpy as np

# ─── MediaPipe setup ──────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(max_num_hands=1,
                          min_detection_confidence=0.7,
                          min_tracking_confidence=0.7)

# ─── Toolbar config ───────────────────────────────────────────────
COLORS = [
    ("Magenta", (255,  0, 255)),
    ("Blue",    (255,  0,   0)),
    ("Green",   (  0,255,   0)),
    ("Red",     (  0,  0, 255)),
    ("Black",   (  0,  0,   0)),
    ("White",   (255,255, 255)),
    ("Purple",  (128,  0,255)),
    ("Yellow",  (  0,255,255)),
    ("Cyan",    (  0,255,  0)),
    ("Orange",  (  0,128,255)), 
]
SIZES = [2, 5, 10, 20, 40]

TOOLBAR_H   = 60          # toolbar height in pixels
COLOR_W     = 60          # width of each color swatch
SIZE_W      = 45          # width of each size button
SIZE_START  = 80          # x where size buttons begin

cur_color = COLORS[0][1]  # default: Magenta
cur_size  = SIZES[1]      # default: 5

# ─── Canvas (drawn on top of camera frame) ────────────────────────
canvas     = None          # initialised after first frame
prev_point = None

def draw_toolbar(frame):
    """Draw the top toolbar onto frame (in-place)."""
    h, w = frame.shape[:2]

    # dark background for toolbar
    cv2.rectangle(frame, (0, 0), (w, TOOLBAR_H), (30, 30, 30), -1)

    # ── SIZE label & buttons ──────────────────────────────────────
    cv2.putText(frame, "SIZE", (5, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    for i, sz in enumerate(SIZES):
        x1 = SIZE_START + i * SIZE_W
        x2 = x1 + SIZE_W - 4
        selected = (sz == cur_size)
        border_col = (0, 230, 255) if selected else (160, 160, 160)
        thick      = 2            if selected else 1
        cv2.rectangle(frame, (x1, 8), (x2, TOOLBAR_H - 8), border_col, thick)
        cv2.putText(frame, str(sz), (x1 + 8, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # ── COLOR swatches ────────────────────────────────────────────
    swatch_start = SIZE_START + len(SIZES) * SIZE_W + 10
    for i, (name, col) in enumerate(COLORS):
        x1 = swatch_start + i * (COLOR_W + 6)
        x2 = x1 + COLOR_W
        cv2.rectangle(frame, (x1, 8), (x2, TOOLBAR_H - 8), col, -1)
        if col == cur_color:
            cv2.rectangle(frame, (x1 - 2, 6), (x2 + 2, TOOLBAR_H - 6),
                          (0, 230, 255), 3)

    return frame


def toolbar_click(x, y, frame_w):
    """Detect toolbar interaction and update globals."""
    global cur_color, cur_size
    if y > TOOLBAR_H:
        return

    # size buttons
    for i, sz in enumerate(SIZES):
        bx1 = SIZE_START + i * SIZE_W
        bx2 = bx1 + SIZE_W - 4
        if bx1 <= x <= bx2:
            cur_size = sz
            return

    # colour swatches
    swatch_start = SIZE_START + len(SIZES) * SIZE_W + 10
    for i, (name, col) in enumerate(COLORS):
        cx1 = swatch_start + i * (COLOR_W + 6)
        cx2 = cx1 + COLOR_W
        if cx1 <= x <= cx2:
            cur_color = col
            return


def fingers_up(lm):
    """Return list of booleans [index, middle] — True if finger is raised."""
    index_up  = lm[8].y  < lm[6].y
    middle_up = lm[12].y < lm[10].y
    return index_up, middle_up


# ─── Main loop ────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("[Air Drawing] Press 'c' to clear canvas | 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]

    # initialise transparent canvas once we know frame size
    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # ── Hand detection ───────────────────────────────────────────
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        lm_list = result.multi_hand_landmarks[0].landmark

        # fingertip pixel coords
        ix = int(lm_list[8].x * w)
        iy = int(lm_list[8].y * h)

        index_up, middle_up = fingers_up(lm_list)

        # draw skeleton on frame
        mp_draw.draw_landmarks(frame,
                               result.multi_hand_landmarks[0],
                               mp_hands.HAND_CONNECTIONS)

        if index_up and not middle_up:
            # ── DRAW MODE ─────────────────────────────────────
            if iy < TOOLBAR_H:
                toolbar_click(ix, iy, w)
                prev_point = None
            else:
                if prev_point:
                    cv2.line(canvas, prev_point, (ix, iy),
                             cur_color, cur_size)
                prev_point = (ix, iy)

            # show fingertip dot
            cv2.circle(frame, (ix, iy), cur_size // 2 + 4,
                       cur_color, -1)

        elif index_up and middle_up:
            # ── MOVE / HOVER MODE (no drawing) ────────────────
            prev_point = None
            cv2.circle(frame, (ix, iy), 12, (200, 200, 200), 2)

        else:
            prev_point = None
    else:
        prev_point = None

    # ── Merge canvas onto frame ──────────────────────────────────
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask     = cv2.threshold(gray_canvas, 1, 255, cv2.THRESH_BINARY)
    mask_inv    = cv2.bitwise_not(mask)
    bg          = cv2.bitwise_and(frame,  frame,  mask=mask_inv)
    fg          = cv2.bitwise_and(canvas, canvas, mask=mask)
    frame       = cv2.add(bg, fg)

    # ── Draw toolbar on top ──────────────────────────────────────
    frame = draw_toolbar(frame)

    cv2.imshow("Air Drawing — OpenCV + MediaPipe", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        print("[Air Drawing] Canvas cleared!")

cap.release()
cv2.destroyAllWindows()
