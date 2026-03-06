from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.utils import timezone
import pymysql
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn import svm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import hashlib
import io
import base64
import os
import time
from django.conf import settings

# ── helpers ──────────────────────────────────────────────────────────────────

def _db():
    return pymysql.connect(
        host='127.0.0.1', port=3306,
        user='root', password='root',
        database='HeartDisease', charset='utf8'
    )

def _ensure_predictions_table():
    """Create predictions history table if it doesn't exist."""
    con = _db()
    with con:
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(150) NOT NULL,
                age INT, sex INT, cp INT, trestbps INT, chol INT,
                fbs INT, restecg INT, thalach INT, exang INT,
                oldpeak FLOAT, slope INT, ca INT, thal INT,
                result TINYINT(1) NOT NULL,
                nb_accuracy FLOAT, svm_accuracy FLOAT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8
        """)
        con.commit()

# ── views ─────────────────────────────────────────────────────────────────────

def index(request):
    if request.method == 'GET':
        return render(request, 'index.html', {})


def Login(request):
    if request.method == 'GET':
        return render(request, 'Login.html', {})


def Register(request):
    if request.method == 'GET':
        return render(request, 'Register.html', {})


def Predict(request):
    if request.method == 'GET':
        return render(request, 'Predict.html', {})


def PredictHeartCondition(request):
    if request.method == 'POST':
        age      = request.POST.get('age', False)
        gender   = request.POST.get('gender', False)
        cp       = request.POST.get('cp', False)
        bps      = request.POST.get('trestbps', False)
        chol     = request.POST.get('chol', False)
        fbs      = request.POST.get('fbs', False)
        ecg      = request.POST.get('restecg', False)
        thalach  = request.POST.get('thalach', False)
        exang    = request.POST.get('exang', False)
        oldpeak  = request.POST.get('oldpeak', False)
        slope    = request.POST.get('slope', False)
        ca       = request.POST.get('ca', False)
        thal     = request.POST.get('thal', False)

        data = 'age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal\n'
        data += age+","+gender+","+cp+","+bps+","+chol+","+fbs+","+ecg+","+thalach+","+exang+","+oldpeak+","+slope+","+ca+","+thal

        testdata_path = os.path.join(settings.BASE_DIR, 'testdata.txt')
        dataset_path  = os.path.join(settings.BASE_DIR, 'dataset.csv')

        with open(testdata_path, 'w') as f:
            f.write(data)

        train = pd.read_csv(dataset_path)
        X = train.values[:, 0:13]
        Y = train.values[:, 13]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

        cls = GaussianNB()
        cls.fit(X, Y)
        y_pred_nb = cls.predict(X_test)
        accuracy = accuracy_score(Y_test, y_pred_nb) * 100

        svm_cls = svm.SVC()
        svm_cls.fit(X, Y)
        y_pred_svm = svm_cls.predict(X_test)
        svm_accuracy = accuracy_score(Y_test, y_pred_svm) * 100

        test = pd.read_csv(testdata_path)
        test = test.values[:, 0:13]
        y_pred_test = cls.predict(test)

        has_disease = any(str(y_pred_test[i]) == '1.0' for i in range(len(test)))

        # Chart
        height = [accuracy, svm_accuracy]
        bars = ('Naive Bayesian', 'SVM')
        y_pos = np.arange(len(bars))
        plt.figure(figsize=(5, 3))
        plt.bar(y_pos, height, color=['#0369a1', '#0891b2'], width=0.4)
        plt.xticks(y_pos, bars)
        plt.ylim(0, 100)
        plt.ylabel('Accuracy (%)')
        plt.title('Model Accuracy Comparison')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120)
        buf.seek(0)
        chart_data = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        # Save prediction to history
        username = request.session.get('username', '')
        if username:
            try:
                _ensure_predictions_table()
                con = _db()
                with con:
                    cur = con.cursor()
                    cur.execute("""
                        INSERT INTO predictions
                          (username, age, sex, cp, trestbps, chol, fbs, restecg,
                           thalach, exang, oldpeak, slope, ca, thal, result,
                           nb_accuracy, svm_accuracy)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """, (username, age, gender, cp, bps, chol, fbs, ecg,
                          thalach, exang, oldpeak, slope, ca, thal,
                          1 if has_disease else 0,
                          round(accuracy, 1), round(svm_accuracy, 1)))
                    con.commit()
            except Exception:
                pass  # History saving is non-critical

        context = {
            'has_disease':   has_disease,
            'nb_accuracy':   round(accuracy, 1),
            'svm_accuracy':  round(svm_accuracy, 1),
            'chart':         chart_data,
            'username':      username,
            # Store inputs for PDF export
            'p_age': age, 'p_sex': gender, 'p_cp': cp, 'p_trestbps': bps,
            'p_chol': chol, 'p_fbs': fbs, 'p_restecg': ecg, 'p_thalach': thalach,
            'p_exang': exang, 'p_oldpeak': oldpeak, 'p_slope': slope,
            'p_ca': ca, 'p_thal': thal,
        }
        # Cache result in session so PDF export can access it
        request.session['last_result'] = {
            'has_disease': has_disease,
            'nb_accuracy': round(accuracy, 1),
            'svm_accuracy': round(svm_accuracy, 1),
            'username': username,
            'age': age, 'sex': gender, 'cp': cp, 'trestbps': bps,
            'chol': chol, 'fbs': fbs, 'restecg': ecg, 'thalach': thalach,
            'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca,
            'thal': thal,
        }
        return render(request, 'Result.html', context)


def Signup(request):
    if request.method == 'POST':
        username = request.POST.get('username', False)
        password = request.POST.get('password', False)
        contact  = request.POST.get('contact', False)
        email    = request.POST.get('email', False)
        address  = request.POST.get('address', False)

        hashed_password = hashlib.sha256(password.encode()).hexdigest()

        db_connection = _db()
        db_cursor = db_connection.cursor()
        sql = "INSERT INTO register(username, password, contact, email, address) VALUES(%s, %s, %s, %s, %s)"
        db_cursor.execute(sql, (username, hashed_password, contact, email, address))
        db_connection.commit()
        if db_cursor.rowcount == 1:
            context = {'success': True}
            return render(request, 'Register.html', context)
        else:
            context = {'error': 'Registration failed. Please try again.'}
            return render(request, 'Register.html', context)
        db_connection.close()


def HeartViewer(request):
    return render(request, 'HeartViewer.html', {})


# ── Login with rate limiting ──────────────────────────────────────────────────

MAX_ATTEMPTS = 5
LOCKOUT_SECONDS = 15 * 60  # 15 minutes

def UserLogin(request):
    if request.method == 'POST':
        # Rate limiting check
        fail_count  = request.session.get('login_fail_count', 0)
        lockout_until = request.session.get('login_lockout_until', 0)

        if lockout_until and time.time() < lockout_until:
            remaining = int((lockout_until - time.time()) / 60) + 1
            context = {'error': f'Too many failed attempts. Please wait {remaining} minute(s) before trying again.'}
            return render(request, 'Login.html', context)

        username = request.POST.get('username', False)
        password = request.POST.get('password', False)
        hashed_password = hashlib.sha256(password.encode()).hexdigest()

        con = _db()
        with con:
            cur = con.cursor()
            cur.execute(
                "SELECT username FROM register WHERE "
                "(username = %s OR email = %s OR contact = %s) AND password = %s",
                (username, username, username, hashed_password)
            )
            row = cur.fetchone()

        if row:
            # Reset rate limiting on success
            request.session.pop('login_fail_count', None)
            request.session.pop('login_lockout_until', None)
            actual_username = row[0]
            request.session['username'] = actual_username
            context = {'username': actual_username}
            return render(request, 'UserScreen.html', context)
        else:
            fail_count += 1
            request.session['login_fail_count'] = fail_count
            if fail_count >= MAX_ATTEMPTS:
                request.session['login_lockout_until'] = time.time() + LOCKOUT_SECONDS
                request.session['login_fail_count'] = 0
                context = {'error': 'Too many failed attempts. Account locked for 15 minutes.'}
            else:
                remaining = MAX_ATTEMPTS - fail_count
                context = {'error': f'Invalid username or password. {remaining} attempt(s) remaining.'}
            return render(request, 'Login.html', context)


# ── Prediction History ────────────────────────────────────────────────────────

def History(request):
    username = request.session.get('username', '')
    if not username:
        return redirect('Login')

    records = []
    try:
        _ensure_predictions_table()
        con = _db()
        with con:
            cur = con.cursor()
            cur.execute(
                "SELECT id, result, nb_accuracy, svm_accuracy, created_at, age, trestbps, chol "
                "FROM predictions WHERE username=%s ORDER BY created_at DESC LIMIT 50",
                (username,)
            )
            rows = cur.fetchall()
        for r in rows:
            records.append({
                'id': r[0],
                'result': r[1],
                'nb_accuracy': r[2],
                'svm_accuracy': r[3],
                'created_at': r[4],
                'age': r[5],
                'trestbps': r[6],
                'chol': r[7],
            })
    except Exception:
        pass

    return render(request, 'History.html', {'username': username, 'records': records})


# ── Password Change ───────────────────────────────────────────────────────────

def ChangePassword(request):
    username = request.session.get('username', '')
    if not username:
        return redirect('Login')

    if request.method == 'GET':
        return render(request, 'ChangePassword.html', {'username': username})

    if request.method == 'POST':
        current  = request.POST.get('current_password', '')
        new_pw   = request.POST.get('new_password', '')
        confirm  = request.POST.get('confirm_password', '')

        if new_pw != confirm:
            return render(request, 'ChangePassword.html', {
                'username': username,
                'error': 'New passwords do not match.'
            })
        if len(new_pw) < 6:
            return render(request, 'ChangePassword.html', {
                'username': username,
                'error': 'New password must be at least 6 characters.'
            })

        hashed_current = hashlib.sha256(current.encode()).hexdigest()
        con = _db()
        with con:
            cur = con.cursor()
            cur.execute(
                "SELECT username FROM register WHERE username=%s AND password=%s",
                (username, hashed_current)
            )
            row = cur.fetchone()

        if not row:
            return render(request, 'ChangePassword.html', {
                'username': username,
                'error': 'Current password is incorrect.'
            })

        hashed_new = hashlib.sha256(new_pw.encode()).hexdigest()
        con = _db()
        with con:
            cur = con.cursor()
            cur.execute(
                "UPDATE register SET password=%s WHERE username=%s",
                (hashed_new, username)
            )
            con.commit()

        return render(request, 'ChangePassword.html', {
            'username': username,
            'success': 'Password changed successfully.'
        })


# ── PDF Export ────────────────────────────────────────────────────────────────

def ExportPDF(request):
    username = request.session.get('username', '')
    if not username:
        return redirect('Login')

    last = request.session.get('last_result')
    if not last:
        return redirect('Predict')

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
    except ImportError:
        return HttpResponse(
            "reportlab is not installed. Run: pip install reportlab",
            content_type='text/plain', status=500
        )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    styles = getSampleStyleSheet()
    title_style  = ParagraphStyle('Title2',  parent=styles['Title'],  fontSize=20, spaceAfter=4)
    sub_style    = ParagraphStyle('Sub',     parent=styles['Normal'], fontSize=10, textColor=colors.grey)
    heading_style= ParagraphStyle('Heading', parent=styles['Heading2'],fontSize=12, spaceBefore=14, spaceAfter=6)
    body_style   = ParagraphStyle('Body',    parent=styles['Normal'], fontSize=10, leading=16)
    result_ok    = ParagraphStyle('ROK',     parent=styles['Normal'], fontSize=14, textColor=colors.HexColor('#22c55e'), fontName='Helvetica-Bold')
    result_bad   = ParagraphStyle('RBAD',    parent=styles['Normal'], fontSize=14, textColor=colors.HexColor('#ef4444'), fontName='Helvetica-Bold')

    from datetime import datetime
    now = datetime.now().strftime('%d %B %Y, %H:%M')

    story = []
    story.append(Paragraph("VitaHeart — Prediction Report", title_style))
    story.append(Paragraph(f"Generated on {now} &nbsp;|&nbsp; Patient: {last['username']}", sub_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#334155'), spaceAfter=10))

    # Result
    story.append(Paragraph("Prediction Result", heading_style))
    if last['has_disease']:
        story.append(Paragraph("Heart Disease DETECTED", result_bad))
        story.append(Spacer(1, 4))
        story.append(Paragraph(
            "The ML model predicts the presence of cardiovascular disease. "
            "Please consult a cardiologist immediately.", body_style))
    else:
        story.append(Paragraph("No Heart Disease Detected", result_ok))
        story.append(Spacer(1, 4))
        story.append(Paragraph(
            "The ML model predicts no cardiovascular disease. "
            "Continue maintaining a heart-healthy lifestyle.", body_style))

    # Accuracy
    story.append(Paragraph("Model Accuracy", heading_style))
    acc_data = [
        ['Algorithm', 'Accuracy'],
        ['Naïve Bayesian', f"{last['nb_accuracy']}%"],
        ['SVM', f"{last['svm_accuracy']}%"],
    ]
    acc_table = Table(acc_data, colWidths=[10*cm, 5*cm])
    acc_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f172a')),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f8fafc'), colors.white]),
        ('GRID',       (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('FONTSIZE',   (0,0), (-1,-1), 10),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING',(0,0),(-1,-1),6),
    ]))
    story.append(acc_table)

    # Clinical inputs
    story.append(Paragraph("Clinical Parameters Entered", heading_style))
    cp_map    = {'0': 'Typical Angina', '1': 'Atypical Angina', '2': 'Non-Anginal Pain', '3': 'Asymptomatic'}
    ecg_map   = {'0': 'Normal', '1': 'ST-T Abnormality', '2': 'LV Hypertrophy'}
    slope_map = {'1': 'Upsloping', '2': 'Flat', '3': 'Downsloping'}
    thal_map  = {'1': 'Normal', '2': 'Fixed Defect', '3': 'Reversible Defect'}
    sex_map   = {'1': 'Male', '0': 'Female'}
    fbs_map   = {'1': 'True (>120 mg/dl)', '0': 'False (≤120 mg/dl)'}
    exang_map = {'1': 'Yes', '0': 'No'}

    params = [
        ['Parameter', 'Value'],
        ['Age', last.get('age', '—')],
        ['Sex', sex_map.get(str(last.get('sex','')), last.get('sex','—'))],
        ['Chest Pain Type', cp_map.get(str(last.get('cp','')), last.get('cp','—'))],
        ['Resting BP (mm Hg)', last.get('trestbps', '—')],
        ['Cholesterol (mg/dl)', last.get('chol', '—')],
        ['Fasting Blood Sugar', fbs_map.get(str(last.get('fbs','')), last.get('fbs','—'))],
        ['Resting ECG', ecg_map.get(str(last.get('restecg','')), last.get('restecg','—'))],
        ['Max Heart Rate (bpm)', last.get('thalach', '—')],
        ['Exercise Angina', exang_map.get(str(last.get('exang','')), last.get('exang','—'))],
        ['ST Depression (oldpeak)', last.get('oldpeak', '—')],
        ['Slope of ST', slope_map.get(str(last.get('slope','')), last.get('slope','—'))],
        ['Major Vessels', last.get('ca', '—')],
        ['Thalassemia', thal_map.get(str(last.get('thal','')), last.get('thal','—'))],
    ]
    p_table = Table(params, colWidths=[9*cm, 6*cm])
    p_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f172a')),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('ALIGN',      (1,0), (1,-1), 'CENTER'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f8fafc'), colors.white]),
        ('GRID',       (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('FONTSIZE',   (0,0), (-1,-1), 9),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING',(0,0),(-1,-1),5),
    ]))
    story.append(p_table)

    # Disclaimer
    story.append(Spacer(1, 18))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#94a3b8')))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<i>Disclaimer: This report is generated by a machine learning model trained on the UCI Heart Disease "
        "dataset (303 records). It is for <b>educational and research purposes only</b> and does not "
        "constitute medical advice. Consult a qualified healthcare professional for clinical decisions.</i>",
        ParagraphStyle('disc', parent=styles['Normal'], fontSize=8, textColor=colors.grey, leading=13)
    ))

    doc.build(story)
    buf.seek(0)
    response = HttpResponse(buf, content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="VitaHeart_Report_{username}.pdf"'
    return response
