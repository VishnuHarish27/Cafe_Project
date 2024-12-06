from flask import Flask, render_template, Response, request, jsonify, send_file
import cv2
from yolo import YOLO
import os
from database import init_db, get_analytics, clear_table, log_violation
import pandas as pd
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import logging

app = Flask(__name__)
app.config['CLIPS_FOLDER'] = 'D:\\CP-BayesVision\\Violation_detection\\clips'
init_db()  # Initialize the database and create the table if it does not exist
yolo = YOLO(use_gpu=True)

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
CLIPS_FOLDER = 'clips'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['CLIPS_FOLDER'] = CLIPS_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(CLIPS_FOLDER, exist_ok=True)

def gen_frames(video_path, save_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    frame_idx = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame, aruco_ids = yolo.detect_and_check(frame, frame_idx)
            if out is None:
                # Initialize the video writer with the same size as the frame
                out = cv2.VideoWriter(save_path, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            out.write(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            frame_idx += 1

    cap.release()
    if out:
        out.release()

    yolo.save_clips(video_path, app.config['CLIPS_FOLDER'])

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



@app.route('/analytics')
def analytics():
    data = get_analytics()

    # Separate ArUco and Cup data
    aruco_data = [row for row in data if row[4] > 0]  # Assuming row[4] is ArUco Count
    cup_data = [row for row in data if row[3] > 0]  # Assuming row[3] is Cup Count

    return render_template('analytics.html',
                           aruco_data=aruco_data,
                           cup_data=cup_data,
                           aruco_data_count=len(aruco_data),
                           cup_data_count=len(cup_data))

@app.route('/analytics/data')
def analytics_data():
    data = get_analytics()
    return jsonify(data)
@app.route('/download/csv')
def download_csv():
    data = get_analytics()
    aruco_data = [row for row in data if row[4] > 0]
    cup_data = [row for row in data if row[3] > 0]

    aruco_df = pd.DataFrame(aruco_data, columns=['S.No', 'Datetime', 'Frame Index', 'Cup Count', 'Aruco Count', 'ArUco IDs'])
    cup_df = pd.DataFrame(cup_data, columns=['S.No', 'Datetime', 'Frame Index', 'Cup Count', 'Aruco Count', 'ArUco IDs'])

    # Select only the relevant columns for the CSV
    aruco_df = aruco_df[['Datetime', 'Frame Index', 'Aruco Count', 'ArUco IDs']]
    cup_df = cup_df[['Datetime', 'Frame Index', 'Cup Count']]

    with BytesIO() as output:
        aruco_df.to_csv(output, index=False)
        output.write(b"\n\n")
        cup_df.to_csv(output, index=False)
        output.seek(0)
        csv_data = output.getvalue()

    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=detected_objects.csv"}
    )

@app.route('/download/excel')
def download_excel():
    data = get_analytics()
    aruco_data = [row for row in data if row[4] > 0]
    cup_data = [row for row in data if row[3] > 0]

    aruco_df = pd.DataFrame(aruco_data, columns=['S.No', 'Datetime', 'Frame Index', 'Cup Count', 'Aruco Count', 'ArUco IDs'])
    cup_df = pd.DataFrame(cup_data, columns=['S.No', 'Datetime', 'Frame Index', 'Cup Count', 'Aruco Count', 'ArUco IDs'])

    # Select only the relevant columns for the Excel file
    aruco_df = aruco_df[['Datetime', 'Frame Index', 'Aruco Count', 'ArUco IDs']]
    cup_df = cup_df[['Datetime', 'Frame Index', 'Cup Count']]

    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    aruco_df.to_excel(writer, index=False, sheet_name='ArUco Data')
    cup_df.to_excel(writer, index=False, sheet_name='Cup Data')
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
    aruco_data = [row for row in data if row[4] > 0]
    cup_data = [row for row in data if row[3] > 0]

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Title
    c.drawString(100, height - 40, "Detected Objects Report")

    # ArUco Data Table
    c.drawString(100, height - 80, "ArUco Data")
    y = height - 100
    for row in aruco_data:
        c.drawString(100, y, f"{row[1]} - {row[2]} - {row[4]} - {row[5]}")
        y -= 20

    # Cup Data Table
    y -= 40
    c.drawString(100, y, "Cup Data")
    y -= 20
    for row in cup_data:
        c.drawString(100, y, f"{row[1]} - {row[2]} - {row[3]}")
        y -= 20

    c.showPage()
    c.save()
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name='detected_objects.pdf')


@app.route('/download/clip/<filename>')
def download_clip(filename):
    clip_path = os.path.abspath(os.path.join(app.config['CLIPS_FOLDER'], filename))  # Ensure absolute path
    logging.debug(f"Looking for file: {clip_path}")
    if not os.path.exists(clip_path):
        logging.error(f"File not found: {clip_path}")
        return "File not found", 404
    logging.debug(f"File found: {clip_path}")
    return send_file(clip_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
