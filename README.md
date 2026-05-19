# 🚀 PSL Sign Language App - Complete Setup Guide

Yar is project ko apne laptop par chalane ke liye neeche diye gaye simple steps follow karo. Hum Alibaba ka fast mirror server use kar rahe hain taake libraries jaldi download ho jayein.

---

# 📂 Step 1: Project Folder Open Karo

VS Code ya Cursor me apna project folder open karo:

sign language detection

Phir upar menu me jao:

Terminal -> New Terminal

Ya keyboard se:

Ctrl + `

---

# 🛠️ Step 2: Virtual Environment (venv) Banao

Terminal me ye command run karo:

    python -m venv .venv

Ye command chalne ke baad project folder ke andar `.venv` naam ka folder ban jayega.

---

# 🔑 Step 3: venv Activate Karo

Ab virtual environment activate karo:

    .venv\Scripts\activate

Agar successfully activate ho gaya to terminal ke start me:

    (.venv)

likha hua nazar aayega.

---

# 📦 Step 4: Required Libraries Install Karo

Ab sari required libraries install karo:

    pip install streamlit opencv-python mediapipe numpy tensorflow -i https://mirrors.aliyun.com/pypi/simple/

---

# 📚 Installed Libraries Ka Kaam

### streamlit
Web app aur dark-themed UI banane ke liye.

### opencv-python
Laptop webcam access aur video frames process karne ke liye.

### mediapipe
Hands, face, aur pose landmarks detect karne ke liye.

### numpy
Numerical arrays aur data processing ke liye.

### tensorflow
`.keras` AI model load aur prediction run karne ke liye.

---

# 🎯 Step 5: App Run Karo

Jab installation complete ho jaye aur terminal me:

    Successfully installed

aa jaye to final command run karo:

    streamlit run app.py

---

# 🌐 Browser Me App Open Hogi

Command run karte hi browser automatically open ho jayega aur app yahan chalegi:

    http://localhost:8501

🎉 PSL Sign Language Detection App successfully run ho jayegi.
