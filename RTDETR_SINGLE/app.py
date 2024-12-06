from flask import Flask, render_template, Response, request, jsonify, send_file
import cv2
from yolo import YOLO  # Consider renaming this import if you change the class name
import os
from database import init_db, get_analytics, clear_table
import pandas as pd
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

app = Flask(__name__)
init_db()  # Initialize the database and create the table if it does not exist
yolo = YOLO(use_gpu=True)

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def gen_frames(video_path, save_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    while True:
        success, frame = cap.read()  # read the frame from the video
        if not success:
            break
        else:
            frame = yolo.detect(frame)  # detect people in the frame using RTDETR
            if out is None:
                # Initialize the video writer with the same size as the frame
                out = cv2.VideoWriter(save_path, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            out.write(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    if out:
        out.release()

@app.route('/video_feed')
def video_feed():
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'video.mp4')
    save_path = os.path.join(app.config['PROCESSED_FOLDER'], 'processed_video.mp4')
    return Response(gen_frames(video_path, save_path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'video' not in request.files:
            return 'No file part'

        file = request.files['video']
        if file.filename == '':
            return 'No selected file'

        if file:
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'video.mp4')
            file.save(video_path)
            clear_table()  # Clear previous detections before starting a new video
            return 'Video uploaded successfully!'

    uploaded_video = os.path.join(app.config['UPLOAD_FOLDER'], 'video.mp4')
    if os.path.exists(uploaded_video):
        return render_template('index.html', video_uploaded=True)
    else:
        return render_template('index.html', video_uploaded=False)

def upload_new_video(video_path):
    # Process the new video
    yolo.process_new_video(video_path)

@app.route('/analytics')
def analytics():
    data = get_analytics()
    return render_template('analytics.html', data=data, data_count=len(data))

@app.route('/analytics/data')
def analytics_data():
    data = get_analytics()
    return jsonify(data)

@app.route('/download/csv')
def download_csv():
    data = get_analytics()
    df = pd.DataFrame(data, columns=['Datetime', 'Direction', 'Count'])
    csv_data = df.to_csv(index=False)
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=detected_objects.csv"}
    )

@app.route('/download/excel')
def download_excel():
    data = get_analytics()
    df = pd.DataFrame(data, columns=['Datetime', 'Direction', 'Count'])
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.close()
    output.seek(0)
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        download_name='detected_objects.xlsx',
        as_attachment=True
    )

@app.route('/download/pdf')
def download_pdf():
    data = get_analytics()
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    c.drawString(100, height - 40, "Detected Objects Report")
    y = height - 80
    for row in data:
        c.drawString(100, y, f"{row[0]}: {row[1]} - {row[2]}")
        y -= 20
        if y < 40:
            c.showPage()
            y = height - 40
    c.save()
    buffer.seek(0)
    return send_file(
    buffer,
    mimetype='application/pdf',
    as_attachment=True,
    download_name='detected_objects.pdf'
)
@app.route('/download/video')
def download_video():
    video_path = os.path.join(app.config['PROCESSED_FOLDER'], 'processed_video.mp4')
    if os.path.exists(video_path):
        return send_file(video_path, as_attachment=True, download_name='processed_video.mp4')
    else:
        return "Processed video not found!", 404

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
