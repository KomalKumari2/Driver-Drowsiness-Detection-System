import cv2
import numpy as np
import mediapipe as mp
import pygame
import time
import random
import math

from drowsiness import eye_aspect_ratio
from yawning   import mouth_aspect_ratio
from voice     import (speak_yawn, speak_distract, speak_quiz, is_speaking_yawn, is_speaking_distract, is_speaking_quiz)
from alarm     import play_alarm, stop_alarm, set_alarm_volume
from chatbot   import start_chatbot, is_chatbot_active, get_reply
from quiz      import chatbot_reply

#  THRESHOLDS & TIMINGS

EAR_THRESHOLD              = 0.15
EAR_HARD_CLOSED_THRESHOLD  = 0.11
EAR_FRAME_LIMIT            = 60
EAR_RECOVERY_STEP          = 4
YAWN_DIFF_START            = 0.06
YAWN_DIFF_END              = 0.03
YAWN_ABS_THRESHOLD         = 0.82
YAWN_FRAME_START           = 8
YAWN_FRAME_END             = 3
YAWN_HEAD_TILT_MAX         = 12
YAWN_HEAD_NOD_MAX          = 18
YAWN_FACE_YAW_MAX          = 0.18
CALIB_FRAMES               = 50
HEAD_TILT_THRESH           = 18
HEAD_TILT_CLEAR            = 12
HEAD_NOD_THRESH            = 20
HEAD_NOD_CLEAR             = 14
DROWSY_HEAD_TILT_MAX       = 14
DROWSY_HEAD_NOD_MAX        = 16

ALARM_LOW_VOL  = 0.25
ALARM_HIGH_VOL = 1.0

YAWN_COOLDOWN     = 6.0
DISTRACT_COOLDOWN = 5.0

AUTO_TARGET_BRIGHTNESS = 138.0
AUTO_SMOOTHING         = 0.18
AUTO_MAX_BRIGHTNESS    = 70
AUTO_MIN_CONTRAST      = 0.85
AUTO_MAX_CONTRAST      = 1.55

#  ALERT MESSAGES

YAWN_MESSAGES = [
    "You are yawning. Please take a short break.",
    "You seem tired. Pull over and rest for a few minutes.",
    "Feeling sleepy? Stop safely and take rest.",
    "Long drive? A 15-minute nap will keep you safe.",
    "Your body is signaling fatigue. Please rest soon.",
    "Yawning detected. Consider stopping at the next rest area.",
]

DISTRACT_MESSAGES = [
    "Eyes on the road! You appear distracted.",
    "Head tilt detected. Please focus on driving.",
    "Keep your eyes forward and stay alert.",
    "Distraction detected. Drive carefully.",
    "Please keep your head straight. Safety first.",
]

#  KEYBOARD QUIZ (fallback when mic not available)

QUIZ_QUESTIONS = [
    {"q": "5 + 3 = ?",   "opt": ["1: 6",  "2: 8",  "3: 10"], "ans": "2"},
    {"q": "10 - 4 = ?",  "opt": ["1: 6",  "2: 5",  "3: 7"],  "ans": "1"},
    {"q": "3 x 3 = ?",   "opt": ["1: 6",  "2: 9",  "3: 12"], "ans": "2"},
    {"q": "15 / 3 = ?",  "opt": ["1: 5",  "2: 6",  "3: 4"],  "ans": "1"},
    {"q": "7 + 6 = ?",   "opt": ["1: 12", "2: 14", "3: 13"], "ans": "3"},
    {"q": "8 x 2 = ?",   "opt": ["1: 14", "2: 16", "3: 18"], "ans": "2"},
    {"q": "20 / 4 = ?",  "opt": ["1: 4",  "2: 6",  "3: 5"],  "ans": "3"},
    {"q": "9 - 3 = ?",   "opt": ["1: 5",  "2: 6",  "3: 7"],  "ans": "2"},
]

#  MEDIAPIPE

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5)

LEFT_EYE  = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]
MOUTH     = [61,81,13,311,291,308,402,14,178,88,95,185]
NOSE_TIP=4; CHIN=152; LEFT_EAR=234; RIGHT_EAR=454

#  STATE

class DriverState:
    def __init__(self): self.reset()
    def reset(self):
        self.ear_counter=0; self.drowsy=False; self.alarm_on=False
        self.mar_calibration=[]; self.baseline_mar=0.0; self.calibrated=False
        self.yawn_counter=0; self.yawning=False; self.prev_yawning=False
        self.last_yawn_time=-999.0
        self.distracted=False; self.prev_distracted=False; self.last_distract_t=-999.0
        self.face_missing_frames=0
        self.chatbot_running=False; self.use_keyboard_quiz=False
        self.quiz_active=False; self.quiz_question=None
        self.quiz_asked=False; self.quiz_wrong_cnt=0
        self.chatbot_msg=""; self.chatbot_msg_time=0.0
        self.auto_adjust=True; self.auto_brightness=0.0; self.auto_contrast=1.0
        self.brightness=0; self.contrast=1.0
        self.session_start=time.time()

state = DriverState()

CLR = {
    "bg":(15,17,26),"panel":(25,28,42),"border":(50,55,80),
    "accent":(0,200,255),"green":(50,220,80),"red":(50,50,230),
    "yellow":(0,220,220),"orange":(0,140,255),"white":(240,240,245),
    "dimwhite":(160,160,175),"teal":(0,190,170),
}

#  HELPERS

def draw_rounded_rect(img,x,y,w,h,r,color,alpha=0.75):
    ov=img.copy()
    cv2.rectangle(ov,(x+r,y),(x+w-r,y+h),color,-1)
    cv2.rectangle(ov,(x,y+r),(x+w,y+h-r),color,-1)
    for cx,cy,a in [(x+r,y+r,180),(x+w-r,y+r,270),(x+r,y+h-r,90),(x+w-r,y+h-r,0)]:
        cv2.ellipse(ov,(cx,cy),(r,r),a,0,90,color,-1)
    cv2.addWeighted(ov,alpha,img,1-alpha,0,img)

def put_text(img,text,x,y,scale=0.55,color=(240,240,245),thick=1,font=cv2.FONT_HERSHEY_DUPLEX):
    cv2.putText(img,text,(x,y),font,scale,color,thick,cv2.LINE_AA)

def put_bold(img,text,x,y,scale=0.65,color=(240,240,245)):
    cv2.putText(img,text,(x,y),cv2.FONT_HERSHEY_DUPLEX,scale,color,2,cv2.LINE_AA)

def draw_bar(img,x,y,w,h,val,mx,clo,chi):
    r=min(1.0,max(0.0,val/mx))
    cv2.rectangle(img,(x,y),(x+w,y+h),(40,42,55),-1)
    bw=int(w*r)
    if bw>0:
        col=tuple(int(clo[i]+r*(chi[i]-clo[i]))for i in range(3))
        cv2.rectangle(img,(x,y),(x+bw,y+h),col,-1)
    cv2.rectangle(img,(x,y),(x+w,y+h),CLR["border"],1)

def adj(frame,b=0,c=1.0): return cv2.convertScaleAbs(frame,alpha=c,beta=b)

def auto_adjust_values(frame):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    mean=float(np.mean(gray)); std=float(np.std(gray))

    target_b=np.clip((AUTO_TARGET_BRIGHTNESS-mean)*0.9,-AUTO_MAX_BRIGHTNESS,AUTO_MAX_BRIGHTNESS)
    target_c=np.clip(1.10+(55.0-std)/120.0,AUTO_MIN_CONTRAST,AUTO_MAX_CONTRAST)
    if mean>175.0: target_c=max(AUTO_MIN_CONTRAST,target_c-0.15)

    state.auto_brightness=(1.0-AUTO_SMOOTHING)*state.auto_brightness+AUTO_SMOOTHING*target_b
    state.auto_contrast=(1.0-AUTO_SMOOTHING)*state.auto_contrast+AUTO_SMOOTHING*target_c
    return int(round(state.auto_brightness)), round(state.auto_contrast,2)

def current_adjustment_values():
    if state.auto_adjust:
        return int(round(state.auto_brightness)), round(state.auto_contrast,2)
    return state.brightness, state.contrast

def head_angles(lms,w,h):
    def pt(i): lm=lms[i]; return np.array([lm.x*w,lm.y*h])
    lep=pt(LEFT_EAR);rep=pt(RIGHT_EAR);nsp=pt(NOSE_TIP);chp=pt(CHIN)
    tilt=math.degrees(math.atan2(rep[1]-lep[1],rep[0]-lep[0]))
    nod=math.degrees(math.atan2(chp[0]-nsp[0],chp[1]-nsp[1]))
    face_mid=(lep+rep)/2.0
    half_face_width=max(np.linalg.norm(rep-lep)/2.0,1.0)
    face_yaw=abs(nsp[0]-face_mid[0])/half_face_width
    return abs(tilt),abs(nod),face_yaw

def wrap(text,n=30):
    words=text.split(); lines=[]; line=""
    for w in words:
        t=(line+" "+w).strip()
        if len(t)<=n: line=t
        else:
            if line: lines.append(line)
            line=w
    if line: lines.append(line)
    return lines

def trigger_kb_quiz():
    state.quiz_question=random.choice(QUIZ_QUESTIONS)
    state.quiz_active=True; state.quiz_asked=False; state.quiz_wrong_cnt=0


def relax_tracking_state():
    state.ear_counter=max(0,state.ear_counter-EAR_RECOVERY_STEP)
    if not state.alarm_on and state.ear_counter<EAR_FRAME_LIMIT:
        state.drowsy=False

    state.yawn_counter=max(0,state.yawn_counter-2)
    state.prev_yawning=state.yawning
    if state.yawn_counter<=YAWN_FRAME_END:
        state.yawning=False

    state.prev_distracted=state.distracted
    state.distracted=False

#  CHATBOT DONE CALLBACK

def on_chatbot_done(passed):
    state.chatbot_running=False
    if passed:
        stop_alarm(); state.alarm_on=False; state.drowsy=False
        state.ear_counter=0; state.quiz_active=False
        state.chatbot_msg="Alarm stopped! Drive safely."
    else:
        state.use_keyboard_quiz=True; trigger_kb_quiz()
        state.chatbot_msg="Use keyboard quiz to stop alarm."
    state.chatbot_msg_time=time.time()

#  DASHBOARD

def draw_dashboard(canvas,frame_bgr,ear,mar,tilt,nod,fps):
    FW,FH=640,480; PX=FW+10; PW=canvas.shape[1]-PX-8

    feed=cv2.resize(frame_bgr,(FW,FH))
    canvas[10:10+FH,8:8+FW]=feed
    cv2.rectangle(canvas,(7,9),(8+FW,11+FH),CLR["border"],2)
    draw_rounded_rect(canvas,PX,8,PW,canvas.shape[0]-16,10,CLR["panel"],0.9)

    row=28
    put_bold(canvas,"DRIVER MONITOR",PX+12,row+20,0.70,CLR["accent"]); row+=44
    cv2.line(canvas,(PX+10,row),(PX+PW-10,row),CLR["border"],1); row+=12

    el=int(time.time()-state.session_start)
    h_,m_,s_=el//3600,(el%3600)//60,el%60
    put_text(canvas,f"Session:{h_:02d}:{m_:02d}:{s_:02d}",PX+12,row+12,0.44,CLR["dimwhite"])
    put_text(canvas,f"FPS:{fps:4.1f}",PX+PW-70,row+12,0.44,CLR["dimwhite"]); row+=26

    # Status
    if state.drowsy: sc,st=CLR["red"],"DROWSY!"
    elif state.yawning: sc,st=CLR["yellow"],"YAWNING"
    elif state.distracted: sc,st=CLR["orange"],"DISTRACTED"
    else: sc,st=CLR["green"],"AWAKE"
    draw_rounded_rect(canvas,PX+10,row,PW-20,38,6,sc,0.25)
    cv2.rectangle(canvas,(PX+10,row),(PX+PW-10,row+38),sc,2)
    (tw,_),_=cv2.getTextSize(st,cv2.FONT_HERSHEY_DUPLEX,0.80,2)
    put_bold(canvas,st,PX+10+(PW-20-tw)//2,row+26,0.80,sc); row+=48

    def mrow(lbl,val,mx,clo,chi,vc=None):
        nonlocal row
        put_text(canvas,lbl,PX+12,row+12,0.42,CLR["dimwhite"])
        draw_bar(canvas,PX+12,row+15,PW-28,9,val,mx,clo,chi)
        put_text(canvas,f"{val:.2f}",PX+PW-44,row+12,0.40,vc or CLR["dimwhite"]); row+=33

    mrow("EAR-Eye",ear,0.35,(30,30,220),(40,210,70),CLR["green"] if ear>EAR_THRESHOLD else CLR["red"])
    mrow("MAR-Mouth",mar,0.6,(40,210,70),(0,220,220))
    mrow("Head Tilt",tilt,40.0,(40,210,70),(0,140,255),CLR["orange"] if tilt>HEAD_TILT_THRESH else CLR["green"])
    mrow("Head Nod",nod,45.0,(40,210,70),(0,140,255),CLR["orange"] if nod>HEAD_NOD_THRESH else CLR["green"])

    put_text(canvas,"Drowsy Ctr",PX+12,row+12,0.42,CLR["dimwhite"])
    draw_bar(canvas,PX+12,row+15,PW-28,9,state.ear_counter,EAR_FRAME_LIMIT,(40,210,70),(30,30,220))
    put_text(canvas,f"{state.ear_counter}/{EAR_FRAME_LIMIT}",PX+PW-52,row+12,0.38,CLR["dimwhite"]); row+=33

    cv2.line(canvas,(PX+10,row),(PX+PW-10,row),CLR["border"],1); row+=8
    put_text(canvas,"BRIGHTNESS/CONTRAST",PX+12,row+12,0.40,CLR["accent"]); row+=16
    adj_b,adj_c=current_adjustment_values()
    mode="AUTO" if state.auto_adjust else "MANUAL"
    put_text(canvas,f"{mode}  B:{adj_b:+d}  C:{adj_c:.2f}x",PX+12,row+12,0.40,CLR["dimwhite"]); row+=14
    put_text(canvas,"A=Auto  F/D=Bright  E/S=Contrast",PX+12,row+12,0.34,(80,80,100)); row+=20

    cv2.line(canvas,(PX+10,row),(PX+PW-10,row),CLR["border"],1); row+=8
    put_text(canvas,"CONTROLS",PX+12,row+12,0.40,CLR["accent"]); row+=16
    for ctrl in ["T=Chatbot  R=Reset  A=Auto","1/2/3=Quiz  ESC=Quit"]:
        put_text(canvas,ctrl,PX+12,row+10,0.34,(80,80,100)); row+=13

    # Chatbot message panel
    if state.chatbot_msg and (time.time()-state.chatbot_msg_time)<6.0:
        cv2.line(canvas,(PX+10,row+4),(PX+PW-10,row+4),CLR["border"],1); row+=10
        put_text(canvas,"ASSISTANT:",PX+12,row+12,0.38,CLR["teal"]); row+=15
        for ln in wrap(state.chatbot_msg,27):
            put_text(canvas,ln,PX+12,row+12,0.34,CLR["white"]); row+=13

    # ── FEED OVERLAYS ─────────────────────────────────────────

    # Chatbot active
    if state.chatbot_running or is_chatbot_active():
        ox,oy=15,15
        draw_rounded_rect(canvas,ox,oy,FW-22,66,8,(0,55,38),0.90)
        cv2.rectangle(canvas,(ox,oy),(ox+FW-22,oy+66),CLR["teal"],2)
        put_bold(canvas,"CHATBOT ACTIVE",ox+16,oy+26,0.68,CLR["teal"])
        put_text(canvas,"Speak into microphone to answer",ox+16,oy+50,0.42,CLR["white"])

    # Keyboard quiz fallback
    elif state.quiz_active and state.quiz_question and state.use_keyboard_quiz:
        q=state.quiz_question; ox,oy=15,FH-218
        draw_rounded_rect(canvas,ox,oy,FW-22,208,8,(8,8,20),0.90)
        cv2.rectangle(canvas,(ox,oy),(ox+FW-22,oy+208),CLR["red"],2)
        put_bold(canvas,"!! ANSWER TO STOP ALARM !!",ox+14,oy+26,0.54,CLR["red"])
        put_text(canvas,"(Mic not available - use keyboard)",ox+14,oy+46,0.38,CLR["dimwhite"])
        cv2.line(canvas,(ox+8,oy+54),(ox+FW-30,oy+54),CLR["border"],1)
        put_bold(canvas,q["q"],ox+14,oy+80,0.78,CLR["white"])
        for i,opt in enumerate(q["opt"]):
            put_text(canvas,opt,ox+24,oy+115+i*34,0.58,CLR["accent"])
        put_text(canvas,"Press  1  /  2  /  3",ox+14,oy+196,0.46,CLR["yellow"])

    # Calibration bar
    if not state.calibrated:
        pct=int(100*len(state.mar_calibration)/CALIB_FRAMES)
        draw_rounded_rect(canvas,12,FH-50,285,38,6,(8,8,20),0.88)
        put_text(canvas,f"Calibrating mouth baseline... {pct}%",20,FH-26,0.44,CLR["yellow"])

    # Distraction banner
    if state.distracted and not (state.chatbot_running or is_chatbot_active()):
        bx,by=8,FH//2-24
        draw_rounded_rect(canvas,bx,by,FW-16,48,8,(0,50,110),0.72)
        cv2.rectangle(canvas,(bx,by),(bx+FW-16,by+48),CLR["orange"],2)
        put_bold(canvas,"HEAD TILT - EYES ON ROAD",bx+20,by+32,0.60,CLR["orange"])

    return canvas

#  DETECTION HANDLERS

def handle_drowsiness(ear,tilt,nod):
    head_down=(tilt>DROWSY_HEAD_TILT_MAX or nod>DROWSY_HEAD_NOD_MAX)
    eyes_deeply_closed=(ear<EAR_HARD_CLOSED_THRESHOLD)
    should_count=(ear<EAR_THRESHOLD) and (eyes_deeply_closed or not head_down)

    if should_count: state.ear_counter+=1
    else: state.ear_counter=max(0,state.ear_counter-EAR_RECOVERY_STEP)

    if state.ear_counter>=EAR_FRAME_LIMIT:
        if not state.drowsy:
            state.drowsy=True; state.alarm_on=True
            play_alarm(ALARM_LOW_VOL if state.yawning else ALARM_HIGH_VOL)
            if not is_chatbot_active() and not state.chatbot_running:
                state.chatbot_running=True; state.use_keyboard_quiz=False
                start_chatbot(on_complete=on_chatbot_done)
    else:
        state.drowsy=False


def handle_yawn(mar,tilt,nod,face_yaw):
    if not state.calibrated: return

    frontal_face=(tilt<=YAWN_HEAD_TILT_MAX and nod<=YAWN_HEAD_NOD_MAX and face_yaw<=YAWN_FACE_YAW_MAX)
    if not frontal_face:
        state.yawn_counter=max(0,state.yawn_counter-3)
        state.prev_yawning=state.yawning
        if state.yawn_counter<=YAWN_FRAME_END:
            state.yawning=False
        return

    diff=mar-state.baseline_mar
    start_level=max(YAWN_ABS_THRESHOLD,state.baseline_mar+YAWN_DIFF_START)
    stop_level=max(YAWN_ABS_THRESHOLD-0.06,state.baseline_mar+YAWN_DIFF_END)
    mouth_wide_open=(mar>=start_level and diff>=YAWN_DIFF_START)

    if mouth_wide_open: state.yawn_counter+=1
    elif mar<=stop_level: state.yawn_counter=max(0,state.yawn_counter-3)
    else: state.yawn_counter=max(0,state.yawn_counter-1)

    state.prev_yawning=state.yawning
    if not state.yawning and state.yawn_counter>=YAWN_FRAME_START:
        state.yawning=True
    elif state.yawning and state.yawn_counter<=YAWN_FRAME_END:
        state.yawning=False

    now=time.time()
    is_new=state.yawning and not state.prev_yawning
    cooldown=(now-state.last_yawn_time)>=YAWN_COOLDOWN

    if state.yawning and (is_new or cooldown) and not is_speaking_yawn():
        msg=random.choice(YAWN_MESSAGES)
        state.chatbot_msg=get_reply("yawning")
        state.chatbot_msg_time=now
        if state.alarm_on:
            set_alarm_volume(ALARM_LOW_VOL)
            speak_yawn(msg,restore_volume=ALARM_HIGH_VOL)
        else:
            speak_yawn(msg)
        state.last_yawn_time=now


def handle_distraction(tilt,nod):
    now=time.time()
    if tilt>HEAD_TILT_THRESH or nod>HEAD_NOD_THRESH:
        state.prev_distracted=state.distracted; state.distracted=True
        is_new=not state.prev_distracted
        cooldown=(now-state.last_distract_t)>=DISTRACT_COOLDOWN
        if (is_new or cooldown) and not is_speaking_distract():
            speak_distract(random.choice(DISTRACT_MESSAGES))
            state.chatbot_msg=get_reply("distracted")
            state.chatbot_msg_time=now
            state.last_distract_t=now
    else:
        state.prev_distracted=state.distracted
        if tilt<HEAD_TILT_CLEAR and nod<HEAD_NOD_CLEAR:
            state.distracted=False


def handle_keyboard_quiz(key):
    if not state.quiz_active or not state.quiz_question or not state.use_keyboard_quiz: return
    if not state.quiz_asked and not is_speaking_quiz():
        q_text=state.quiz_question["q"].replace("?","")
        speak_quiz(f"Answer to stop alarm. {q_text}. Options: {', '.join(state.quiz_question['opt'])}")
        state.quiz_asked=True

    ans=None
    if key==ord('1'): ans="1"
    elif key==ord('2'): ans="2"
    elif key==ord('3'): ans="3"

    if ans:
        if ans==state.quiz_question["ans"]:
            speak_quiz("Correct! Alarm stopped. Stay alert.")
            stop_alarm(); state.alarm_on=False; state.quiz_active=False
            state.quiz_question=None; state.use_keyboard_quiz=False
            state.ear_counter=0; state.drowsy=False
            state.chatbot_msg="Correct! Alarm stopped."; state.chatbot_msg_time=time.time()
        else:
            state.quiz_wrong_cnt+=1; speak_quiz("Wrong answer. Try again.")
            if state.quiz_wrong_cnt>=2:
                state.quiz_question=random.choice(QUIZ_QUESTIONS)
                state.quiz_asked=False; state.quiz_wrong_cnt=0

#  MAIN

def main():
    cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    cap.set(cv2.CAP_PROP_FPS,30)
    for _ in range(3): cap.read()

    CANVAS_W,CANVAS_H=960,500
    fps_timer=time.time(); fps_count=0; fps=0.0

    print(__doc__)
    print(">> Camera ready. Tip: Press T to test the chatbot.")

    frame=np.zeros((480,640,3),dtype=np.uint8)

    while True:
        ret,raw=cap.read()
        if not ret: continue

        fps_count+=1; now=time.time()
        if now-fps_timer>=1.0:
            fps=fps_count/(now-fps_timer); fps_count=0; fps_timer=now

        raw=cv2.flip(raw,1)
        if state.auto_adjust:
            adj_b,adj_c=auto_adjust_values(raw)
        else:
            adj_b,adj_c=state.brightness,state.contrast
        frame=adj(raw,adj_b,adj_c)
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        result=face_mesh.process(rgb)

        ear=mar=tilt=nod=face_yaw=0.0

        if result.multi_face_landmarks:
            state.face_missing_frames=0
            for fl in result.multi_face_landmarks:
                h,w,_=frame.shape; lms=fl.landmark
                for lm in lms:
                    cv2.circle(frame,(int(lm.x*w),int(lm.y*h)),1,(200,200,210),-1)

                def pts(idx):
                    return np.array([[int(lms[i].x*w),int(lms[i].y*h)]for i in idx],dtype=np.float32)

                ear=(eye_aspect_ratio(pts(LEFT_EYE))+eye_aspect_ratio(pts(RIGHT_EYE)))/2.0
                mar=mouth_aspect_ratio(pts(MOUTH))
                tilt,nod,face_yaw=head_angles(lms,w,h)

                if not state.calibrated:
                    state.mar_calibration.append(mar)
                    if len(state.mar_calibration)>=CALIB_FRAMES:
                        state.baseline_mar=sum(state.mar_calibration)/len(state.mar_calibration)
                        state.calibrated=True
                        print(f"[Calib] MAR baseline={state.baseline_mar:.4f}")

                handle_drowsiness(ear,tilt,nod)
                handle_yawn(mar,tilt,nod,face_yaw)
                handle_distraction(tilt,nod)
        else:
            state.face_missing_frames+=1
            if state.face_missing_frames>=4:
                relax_tracking_state()

        key=cv2.waitKey(1)&0xFF
        handle_keyboard_quiz(key)

        if key in [27,ord('q'),ord('Q')]: break
        elif key in [ord('a'),ord('A')]:
            state.auto_adjust=not state.auto_adjust
            state.chatbot_msg=f"Auto brightness {'enabled' if state.auto_adjust else 'disabled'}."
            state.chatbot_msg_time=time.time()
        elif key in [ord('f'),ord('F')]:
            state.auto_adjust=False; state.brightness=min(100,state.brightness+10)
        elif key in [ord('d'),ord('D')]:
            state.auto_adjust=False; state.brightness=max(-100,state.brightness-10)
        elif key in [ord('e'),ord('E')]:
            state.auto_adjust=False; state.contrast=min(2.5,round(state.contrast+0.1,2))
        elif key in [ord('s'),ord('S')]:
            state.auto_adjust=False; state.contrast=max(0.5,round(state.contrast-0.1,2))
        elif key in [ord('r'),ord('R')]:
            state.ear_counter=0; state.drowsy=False
            state.mar_calibration=[]; state.baseline_mar=0.0; state.calibrated=False
            state.yawn_counter=0; state.yawning=False; state.prev_yawning=False; state.last_yawn_time=-999.0
            state.distracted=False; state.prev_distracted=False; state.last_distract_t=-999.0
            state.face_missing_frames=0
            state.auto_brightness=0.0; state.auto_contrast=1.0
            state.chatbot_msg="Calibration reset."; state.chatbot_msg_time=time.time()
            print("[Reset] Recalibrating...")
        elif key in [ord('t'),ord('T')]:
            if not is_chatbot_active() and not state.chatbot_running:
                play_alarm(0.4); state.alarm_on=True; state.chatbot_running=True
                start_chatbot(on_complete=on_chatbot_done)

        canvas=np.full((CANVAS_H,CANVAS_W,3),CLR["bg"],dtype=np.uint8)
        fh=CANVAS_H-20; fw=int(fh*frame.shape[1]/frame.shape[0])
        fw=min(fw,CANVAS_W-240); fh=int(fw*frame.shape[0]/frame.shape[1])
        draw_dashboard(canvas,cv2.resize(frame,(fw,fh)),ear,mar,tilt,nod,fps)
        cv2.imshow("Driver Drowsiness Monitor",canvas)

    cap.release(); cv2.destroyAllWindows(); stop_alarm()
    print("\n[System] Session ended.")

if __name__=="__main__": main()
