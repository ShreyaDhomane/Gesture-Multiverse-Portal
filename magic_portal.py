import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
import math

# -------------------- LOAD MODEL --------------------
model_path = "hand_landmarker.task"

base_options = mp.tasks.BaseOptions(model_asset_path=model_path)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)

landmarker = vision.HandLandmarker.create_from_options(options)

# -------------------- LOAD WORLDS --------------------
world_images = [
    cv2.imread("worlds/space.jpg"),
    cv2.imread("worlds/forest.jpeg"),
    cv2.imread("worlds/ocean.jpg"),
    cv2.imread("worlds/fire.jpg"),
    cv2.imread("worlds/city.jpg")
]

current_world = 0

# -------------------- VARIABLES --------------------
trail_points = []
portal_open = False

prev_x = None
swipe_cooldown = 0
angle_offset = 0

# -------------------- CIRCLE DETECTION --------------------
def is_circle(trail):
    if len(trail) < 50:
        return False

    x_coords = [p[0] for p in trail]
    y_coords = [p[1] for p in trail]

    center_x = sum(x_coords) / len(x_coords)
    center_y = sum(y_coords) / len(y_coords)

    distances = [np.hypot(p[0]-center_x, p[1]-center_y) for p in trail]
    avg_radius = sum(distances) / len(distances)

    if avg_radius < 50:
        return False

    good_points = sum(abs(d - avg_radius) < 30 for d in distances)
    if good_points < len(trail) * 0.65:
        return False

    start = trail[0]
    end = trail[-1]
    close_dist = np.hypot(end[0] - start[0], end[1] - start[1])

    if close_dist > 80:
        return False

    return True

# -------------------- PORTAL WORLD OVERLAY --------------------
def overlay_world(frame, world_img, center, radius):
    world_resized = cv2.resize(world_img, (2*radius, 2*radius))

    mask = np.zeros((2*radius, 2*radius), dtype=np.uint8)
    cv2.circle(mask, (radius, radius), radius, 255, -1)

    portal_area = frame[
        center[1]-radius:center[1]+radius,
        center[0]-radius:center[0]+radius
    ]

    if portal_area.shape[:2] != world_resized.shape[:2]:
        return frame

    world_masked = cv2.bitwise_and(world_resized, world_resized, mask=mask)
    bg_masked = cv2.bitwise_and(portal_area, portal_area, mask=cv2.bitwise_not(mask))

    frame[
        center[1]-radius:center[1]+radius,
        center[0]-radius:center[0]+radius
    ] = cv2.add(world_masked, bg_masked)

    return frame

# -------------------- FIRE RING 🔥 --------------------
def draw_fire_ring(frame, center, radius, angle_offset):
    sparks = 70
    for i in range(sparks):
        angle = (2 * math.pi / sparks) * i + angle_offset
        r = radius + np.random.randint(-10, 10)
        x = int(center[0] + r * math.cos(angle))
        y = int(center[1] + r * math.sin(angle))
        color = (
            np.random.randint(0, 60),
            np.random.randint(120, 200),
            np.random.randint(200, 255)
        )
        size = np.random.randint(2, 5)
        cv2.circle(frame, (x, y), size, color, -1)
    return frame

# -------------------- SPARKLES WHILE DRAWING --------------------
def draw_trail_sparkles(frame, trail):
    for point in trail[-15:]:
        x, y = point
        for _ in range(3):
            dx = np.random.randint(-6, 6)
            dy = np.random.randint(-6, 6)
            spark_x = x + dx
            spark_y = y + dy
            color = (
                np.random.randint(0, 80),
                np.random.randint(120, 200),
                np.random.randint(200, 255)
            )
            size = np.random.randint(1, 4)
            cv2.circle(frame, (spark_x, spark_y), size, color, -1)

# -------------------- DOCTOR STRANGE STYLE SHIELD --------------------
def draw_doctor_strange_shield(frame, center, radius, angle_offset):
    num_rings = 3
    num_sparks = 100
    num_symbols = 12

    # Multiple concentric rings
    for i in range(num_rings):
        r = radius + i*15
        overlay = frame.copy()
        cv2.circle(overlay, center, r, (255, 215, 0), 2)  # gold/yellow ring
        alpha = 0.3 - i*0.05
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Draw "runes" or symbols around ring
        for j in range(num_symbols):
            angle = (2*np.pi/num_symbols)*j + angle_offset + i*0.2
            x = int(center[0] + r*np.cos(angle))
            y = int(center[1] + r*np.sin(angle))
            cv2.putText(frame, "*", (x-5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Sparks / particles around shield
    for _ in range(num_sparks):
        angle = np.random.uniform(0, 2*np.pi)
        r = radius + np.random.randint(-10, 10)
        x = int(center[0] + r*np.cos(angle))
        y = int(center[1] + r*np.sin(angle))
        color = (255, np.random.randint(150, 255), 0)  # orange/gold
        size = np.random.randint(1, 3)
        cv2.circle(frame, (x, y), size, color, -1)

# -------------------- FIST ✊ --------------------
def is_fist(hand):
    tips = [8, 12, 16, 20]
    palm = hand[0]
    closed = 0
    for tip in tips:
        dist = np.hypot(hand[tip].x - palm.x, hand[tip].y - palm.y)
        if dist < 0.10:
            closed += 1
    return closed >= 3

# -------------------- PALM ✋ --------------------
def is_palm_open(hand):
    tips = [8, 12, 16, 20]
    palm = hand[0]
    open_fingers = 0
    for tip in tips:
        dist = np.hypot(hand[tip].x - palm.x, hand[tip].y - palm.y)
        if dist > 0.22:
            open_fingers += 1
    return open_fingers >= 3

# -------------------- WEBCAM --------------------
cap = cv2.VideoCapture(0)
frame_id = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_id += 1

    if swipe_cooldown > 0:
        swipe_cooldown -= 1

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )

    result = landmarker.detect_for_video(mp_image, frame_id * 33)

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]
        index_tip = hand[8]
        x = int(index_tip.x * w)
        y = int(index_tip.y * h)

        # ---------------- PORTAL CLOSED ----------------
        if not portal_open:
            trail_points.append((x, y))

            # Doctor Strange style shield
            if is_palm_open(hand):
                draw_doctor_strange_shield(frame, (w//2, h//2), 160, angle_offset)
                cv2.putText(frame, "SHIELD", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
                angle_offset += 0.05  # rotate shield

            # Portal open when circle complete
            if len(trail_points) > 60 and is_circle(trail_points):
                portal_open = True
                trail_points.clear()
                prev_x = None
                print("🌀 Portal Opened!")

        # ---------------- PORTAL OPEN ----------------
        else:
            if prev_x is None:
                prev_x = x

            # Swipe changes world
            if swipe_cooldown == 0 and (x - prev_x) > 90:
                current_world = (current_world + 1) % len(world_images)
                swipe_cooldown = 15
                print("👉 World Changed!")

            prev_x = x

            # Close portal
            if is_fist(hand):
                portal_open = False
                trail_points.clear()
                print("✊ Portal Closed!")

    else:
        if not portal_open:
            trail_points.clear()

    # Draw trail line
    for i in range(1, len(trail_points)):
        cv2.line(frame, trail_points[i-1], trail_points[i], (255, 255, 255), 2)

    # Sparkles while drawing
    if not portal_open:
        draw_trail_sparkles(frame, trail_points)

    # Draw portal
    if portal_open:
        center = (w//2, h//2)
        radius = 110

        frame = overlay_world(frame, world_images[current_world], center, radius)
        angle_offset += 0.15
        frame = draw_fire_ring(frame, center, radius + 5, angle_offset)
        cv2.circle(frame, center, radius, (255, 255, 255), 2)
        cv2.putText(frame, "MULTIVERSE PORTAL", (center[0]-170, center[1]-140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Gesture Multiverse Portal", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
