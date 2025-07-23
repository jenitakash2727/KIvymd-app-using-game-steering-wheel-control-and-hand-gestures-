from kivy.core.window import Window
from kivymd.app import MDApp
from kivy.lang import Builder
from kivymd.uix.screen import MDScreen
from kivy.uix.screenmanager import ScreenManager
from kivy.uix.textinput import TextInput
from kivymd.uix.label import MDLabel
import pickle
import pyautogui
import cv2
import mediapipe as mp
import pyttsx3
import numpy as np
import imutils
from imutils.video import VideoStream
import directkeys

Window.size = (310, 500)

screen_helper = """
ScreenManager:
    Onboarding:
    SignScreen:
    LoginScreen:
    Hand:
    Steering:
    ProfileScreen:

<Onboarding>:
    name: "boarding"
    MDFloatLayout:
        md_bg_color: 1,1,1,1
        Image:
            source: "onboarding.jpeg"
            size_hint: 1,1
            pos_hint: {"center_x": 0.5, "center_y": 0.78}
        Button:
            text: "Skip"
            font_size: "18sp"
            size_hint: .4, .08
            pos_hint: {"center_x": .72, "center_y": .3}
            background_color: 0,0,0,0
            color: 1,1,1,1
            on_press: root.manager.current = "sign"
            canvas.before:
                Color:
                    rgb: 11/255, 255/255, 235/255
                RoundedRectangle:
                    size: self.size
                    pos: self.pos
                    radius: [22]

<SignScreen>:
    name: "sign"
    MDFloatLayout:
        md_bg_color: 1,1,1,1
        TextInput:
            id: su
            hint_text: "Email"
            size_hint: .85, .08
            pos_hint: {"center_x": 0.5, "center_y": .52}
            multiline: False
        TextInput:
            id: sp
            hint_text: "Password"
            size_hint: .85, .08
            pos_hint: {"center_x": 0.5, "center_y": .41}
            multiline: False
        Button:
            text: "SIGN"
            size_hint: .4, .08
            pos_hint: {"center_x": .29, "center_y": .3}
            on_press: root.signup()
        Button:
            text: "LOGIN"
            size_hint: .4, .08
            pos_hint: {"center_x": .72, "center_y": .3}

<LoginScreen>:
    name: "Login"
    MDFloatLayout:
        md_bg_color: 1,1,1,1
        TextInput:
            id: lu
            hint_text: "Email"
            size_hint: .85, .08
            pos_hint: {"center_x": 0.5, "center_y": .52}
            multiline: False
        TextInput:
            id: lp
            hint_text: "Password"
            size_hint: .85, .08
            pos_hint: {"center_x": 0.5, "center_y": .41}
            multiline: False
        Button:
            text: "LOGIN"
            size_hint: .4, .08
            pos_hint: {"center_x": .72, "center_y": .3}
            on_press: root.Login()

<Hand>:
    name: "hand"
    MDFloatLayout:
        MDLabel:
            text: "Hand gesture control: 1-right, 2-left, 3-space"
            pos_hint: {"center_x": 0.5, "center_y": 0.2}
        Button:
            text: "Start"
            pos_hint: {"center_x": 0.5, "center_y": 0.1}
            on_press: app.Mediaplayercontrol()

<Steering>:
    name: "steering"
    MDFloatLayout:
        Button:
            text: "Start Steering"
            pos_hint: {"center_x": 0.5, "center_y": 0.1}
            on_press: app.Steeringcontrol()

<ProfileScreen>:
    name: "profile"
    MDFloatLayout:
        Button:
            text: "Media Pl"
            pos_hint: {"center_x": .72, "center_y": .60}
            on_press: app.Mediaplayercontrol()
        Button:
            text: "Volume"
            pos_hint: {"center_x": .2, "center_y": .60}
            on_press: app.Volumecontrol()
        Button:
            text: "Steering"
            pos_hint: {"center_x": .5, "center_y": .1}
            on_press: app.Steeringcontrol()
"""

class Onboarding(MDScreen):
    pass

class SignScreen(MDScreen):
    def signup(self):
        username = self.ids.su.text
        password = self.ids.sp.text
        if username and password:
            pickle.dump(username, open("signusername", "wb"))
            pickle.dump(password, open("signpassword", "wb"))
            self.manager.current = "Login"

class LoginScreen(MDScreen):
    def Login(self):
        username = self.ids.lu.text
        password = self.ids.lp.text
        saved_user = pickle.load(open("signusername", "rb"))
        saved_pass = pickle.load(open("signpassword", "rb"))
        if username == saved_user and password == saved_pass:
            self.manager.current = "profile"

class Hand(MDScreen):
    pass

class Steering(MDScreen):
    pass

class ProfileScreen(MDScreen):
    pass

class MobileApp(MDApp):
    def build(self):
        return Builder.load_string(screen_helper)

    def Mediaplayercontrol(self):
        def count_fingers(hand_landmarks):
            cnt = 0
            thresh = (hand_landmarks.landmark[0].y * 100 - hand_landmarks.landmark[9].y * 100) / 2
            if (hand_landmarks.landmark[5].y * 100 - hand_landmarks.landmark[8].y * 100) > thresh:
                cnt += 1
            if (hand_landmarks.landmark[9].y * 100 - hand_landmarks.landmark[12].y * 100) > thresh:
                cnt += 1
            if (hand_landmarks.landmark[5].x * 100 - hand_landmarks.landmark[4].x * 100) > thresh:
                cnt += 1
            return cnt

        cap = cv2.VideoCapture(0)
        hands = mp.solutions.hands.Hands(max_num_hands=1)
        drawing = mp.solutions.drawing_utils
        prev = -1

        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                cnt = count_fingers(hand)
                if cnt != prev:
                    if cnt == 1:
                        pyautogui.press("right")
                    elif cnt == 2:
                        pyautogui.press("left")
                    elif cnt == 3:
                        pyautogui.press("space")
                    prev = cnt
                drawing.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
            cv2.imshow("Media Control", frame)
            if cv2.waitKey(1) == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def Volumecontrol(self):
        def count_fingers(hand_landmarks):
            cnt = 0
            thresh = (hand_landmarks.landmark[0].y * 100 - hand_landmarks.landmark[9].y * 100) / 2
            if (hand_landmarks.landmark[5].y * 100 - hand_landmarks.landmark[8].y * 100) > thresh:
                cnt += 1
            if (hand_landmarks.landmark[9].y * 100 - hand_landmarks.landmark[12].y * 100) > thresh:
                cnt += 1
            return cnt

        cap = cv2.VideoCapture(0)
        hands = mp.solutions.hands.Hands(max_num_hands=1)
        drawing = mp.solutions.drawing_utils
        prev = -1

        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                cnt = count_fingers(hand)
                if cnt != prev:
                    if cnt == 1:
                        pyautogui.press("volumeup")
                    elif cnt == 2:
                        pyautogui.press("volumedown")
                    prev = cnt
                drawing.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
            cv2.imshow("Volume Control", frame)
            if cv2.waitKey(1) == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def Steeringcontrol(self):
        cam = VideoStream(src=0).start()
        engine = pyttsx3.init()
        engine.say("Start")
        engine.runAndWait()

        while True:
            img = cam.read()
            img = np.flip(img, axis=1)
            img = imutils.resize(img, width=640)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            blurred = cv2.GaussianBlur(hsv, (11, 11), 0)
            lower = np.array([150, 114, 39])
            upper = np.array([180, 255, 255])
            mask = cv2.inRange(blurred, lower, upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

            h, w = img.shape[:2]
            up = mask[0:h // 2, :]
            down = mask[3*h//4:h, 2*w//5:3*w//5]
            cnts_up = imutils.grab_contours(cv2.findContours(up, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))
            cnts_down = imutils.grab_contours(cv2.findContours(down, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))

            if len(cnts_up) > 0:
                c = max(cnts_up, key=cv2.contourArea)
                M = cv2.moments(c)
                cX = int(M["m10"] / (M["m00"] + 1e-5))
                if cX < (w // 2 - 35):
                    directkeys.press_key("w")
                elif cX > (w // 2 + 35):
                    directkeys.press_key("a")

            if len(cnts_down) > 0:
                directkeys.press_key("d")

            cv2.imshow("Steering", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

MobileApp().run()
