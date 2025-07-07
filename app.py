# app.py

import os
import io
import json
import zipfile
from datetime import datetime, timedelta
from flask import Flask, request, render_template, jsonify, send_file
import pandas as pd
import numpy as np
from scipy.signal import cwt, morlet2, butter, filtfilt
import matplotlib
import requests

matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
data_storage = pd.DataFrame(columns=['timestamp', 'accel_x', 'accel_y', 'accel_z', 'temperature', 'humidity'])
ANOMALY_LOG_DIR = "anomalies"
STATIC_DIR = "static"
os.makedirs(ANOMALY_LOG_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

TELEGRAM_TOKEN = "8170227011:AAGdBAGjw2eaoeeSzIVdpfPkyCQ9cGjy--Q" 
TELEGRAM_CHAT_ID = "1290535605"

def send_telegram_notification(message):
    """Mengirim pesan ke channel atau user Telegram melalui bot."""
    api_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'Markdown'
    }
    
    try:
        response = requests.post(api_url, json=payload)
        print(f"Telegram Notification Sent! Status: {response.status_code}")
    except Exception as e:
        print(f"Error sending Telegram notification: {e}")


def lowpass_filter(data, cutoff=10, fs=50, order=4):
    if len(data) < order * 2: return data
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def analyze_axis(data_series, axis_name, sampling_rate=50):
    if len(data_series) < 64:
        return {"cwt_data": [], "max_amplitude": None, "dominant_freq": None, "condition_assessment": "Data Tidak Cukup", "cwt_matrix": None}
    raw_signal = data_series[-128:].to_numpy()
    signal = lowpass_filter(raw_signal, fs=sampling_rate); signal = signal - np.mean(signal)
    widths = np.arange(1, 64); coef = cwt(signal, morlet2, widths, w=5.0); cwt_matrix = np.abs(coef)
    amplitudes = np.max(cwt_matrix, axis=1); cwt_data = [[int(w), float(a)] for w, a in zip(widths, amplitudes)]
    dominant_index = int(np.argmax(amplitudes)); dominant_scale = widths[dominant_index]; dominant_freq = sampling_rate / dominant_scale
    max_amplitude = float(np.max(amplitudes))
    if max_amplitude <= 2000: condition = "Baik (Utuh)"
    elif max_amplitude <= 5000: condition = "Cukup (Rusak Ringan Non-Struktural)"
    elif max_amplitude <= 15000: condition = "Sedang (Rusak Ringan Struktural)"
    else: condition = "Buruk (Rusak Berat)"
    plt.figure(figsize=(8, 4)); plt.imshow(cwt_matrix, extent=[0, len(signal), widths[-1], widths[0]], cmap='jet', aspect='auto')
    plt.colorbar(label='Amplitude'); plt.xlabel("Time Step"); plt.ylabel("Scale"); plt.title(f"CWT Scalogram (Axis {axis_name.upper()})"); plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, f"scalogram_{axis_name}.png")); plt.close()
    return {"cwt_data": cwt_data, "max_amplitude": max_amplitude, "dominant_freq": round(dominant_freq, 2), "condition_assessment": condition, "cwt_matrix": cwt_matrix}


@app.route("/")
def index(): return render_template("index.html")

@app.route("/data", methods=["POST"])
def receive_data():
    global data_storage
    try:
        data = request.get_json(force=True)
        new_data = pd.DataFrame([{"timestamp": datetime.now(), "accel_x": int(data.get("accel_x")), "accel_y": int(data.get("accel_y")), "accel_z": int(data.get("accel_z")), "temperature": float(data.get("temperature")), "humidity": float(data.get("humidity"))}])
        data_storage = pd.concat([data_storage, new_data], ignore_index=True)
        return jsonify({"status": "success"})
    except Exception as e: return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/api/latest")
def latest_data():
    if len(data_storage) < 128: return jsonify({"error": "Data belum cukup untuk analisis"}), 404
    analysis_x = analyze_axis(data_storage['accel_x'], 'x')
    analysis_y = analyze_axis(data_storage['accel_y'], 'y')
    analysis_z = analyze_axis(data_storage['accel_z'], 'z')
    
    conditions = [analysis_x['condition_assessment'], analysis_y['condition_assessment'], analysis_z['condition_assessment']]
    is_anomaly = any(cond in ["Sedang (Rusak Ringan Struktural)", "Buruk (Rusak Berat)"] for cond in conditions)

    if is_anomaly:
        event_timestamp = datetime.now()
        worst_condition = "Sedang (Rusak Ringan Struktural)"
        if "Buruk (Rusak Berat)" in conditions:
            worst_condition = "Buruk (Rusak Berat)"

        message = (
            f"🚨 *ANOMALI GETARAN TERDETEKSI* 🚨\n\n"
            f"*Waktu:* {event_timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"*Level Kondisi:* *{worst_condition}*\n\n"
            f"Segera periksa dashboard atau unduh log anomali untuk detail."
        )
        send_telegram_notification(message)

        # Logika penyimpanan file anomali tetap berjalan
        data_block = data_storage.tail(128).copy()
        data_block['timestamp'] = data_block['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S.%f')
        anomaly_data = {
            "event_timestamp": event_timestamp.isoformat(),
            "analysis_summary": {
                "x": {"condition": analysis_x["condition_assessment"], "dominant_freq": analysis_x["dominant_freq"], "max_amplitude": analysis_x["max_amplitude"]},
                "y": {"condition": analysis_y["condition_assessment"], "dominant_freq": analysis_y["dominant_freq"], "max_amplitude": analysis_y["max_amplitude"]},
                "z": {"condition": analysis_z["condition_assessment"], "dominant_freq": analysis_z["dominant_freq"], "max_amplitude": analysis_z["max_amplitude"]},
            },
            "raw_data": data_block[['timestamp', 'accel_x', 'accel_y', 'accel_z']].to_dict('records')
        }
        filename = f"anomaly_{event_timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(ANOMALY_LOG_DIR, filename)
        with open(filepath, 'w') as f: json.dump(anomaly_data, f, indent=4)
        print(f"🚨 Anomali Komprehensif terdeteksi! Data disimpan dan notifikasi dikirim.")

    # Sisa dari fungsi ini tidak berubah
    latest = data_storage.iloc[-1]
    one_min_ago = datetime.now() - timedelta(minutes=1)
    recent = data_storage[data_storage['timestamp'] > one_min_ago]
    for analysis in [analysis_x, analysis_y, analysis_z]: analysis.pop('cwt_matrix', None)
    return jsonify({
        "accel_x": int(latest['accel_x']), "accel_y": int(latest['accel_y']), "accel_z": int(latest['accel_z']),
        "temperature": float(latest['temperature']), "humidity": float(latest['humidity']),
        "chart_data_x": {"timestamps": recent['timestamp'].dt.strftime("%H:%M:%S").tolist(), "values": recent['accel_x'].tolist()},
        "chart_data_y": {"timestamps": recent['timestamp'].dt.strftime("%H:%M:%S").tolist(), "values": recent['accel_y'].tolist()},
        "chart_data_z": {"timestamps": recent['timestamp'].dt.strftime("%H:%M:%S").tolist(), "values": recent['accel_z'].tolist()},
        "analysis_x": analysis_x, "analysis_y": analysis_y, "analysis_z": analysis_z
    })

# --- Sisa endpoint tidak berubah ---
@app.route("/downloads")
def downloads_page(): return render_template("downloads.html")

@app.route("/api/download")
def download_data():
    if len(data_storage) < 128: return "Data tidak cukup untuk diunduh (min. 128)", 404
    data_block = data_storage.tail(128).copy()
    analysis_x = analyze_axis(data_block['accel_x'], 'x'); analysis_y = analyze_axis(data_block['accel_y'], 'y'); analysis_z = analyze_axis(data_block['accel_z'], 'z')
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('timeseries.csv', data_block.to_csv(index=False))
        summary_df = pd.DataFrame({'Sumbu': ['X', 'Y', 'Z'], 'Frekuensi_Dominan_Hz': [a['dominant_freq'] for a in [analysis_x, analysis_y, analysis_z]], 'Amplitudo_Maks_CWT': [a['max_amplitude'] for a in [analysis_x, analysis_y, analysis_z]], 'Kondisi': [a['condition_assessment'] for a in [analysis_x, analysis_y, analysis_z]]})
        zf.writestr('analysis_summary.csv', summary_df.to_csv(index=False))
        for axis, analysis in [('x', analysis_x), ('y', analysis_y), ('z', analysis_z)]:
            if analysis['cwt_matrix'] is not None: zf.writestr(f'cwt_spectrum_{axis}.csv', pd.DataFrame(analysis['cwt_matrix']).to_csv())
    memory_file.seek(0)
    return send_file(memory_file, download_name=f'vibration_snapshot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip', as_attachment=True)

@app.route("/api/anomalies")
def list_anomalies():
    try:
        files = sorted([f for f in os.listdir(ANOMALY_LOG_DIR) if f.endswith('.json')], reverse=True)
        return jsonify(files)
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route("/anomalies/<filename>")
def download_anomaly_file(filename): return send_file(os.path.join(ANOMALY_LOG_DIR, filename), as_attachment=True)

@app.route('/api/anomalies/<filename>', methods=['DELETE'])
def delete_anomaly_file(filename):
    try:
        filepath = os.path.join(ANOMALY_LOG_DIR, filename)
        if not os.path.abspath(filepath).startswith(os.path.abspath(ANOMALY_LOG_DIR)): return jsonify({"status": "error", "message": "Akses ditolak"}), 403
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({"status": "success", "message": f"File {filename} berhasil dihapus."})
        else: return jsonify({"status": "error", "message": "File tidak ditemukan."}), 404
    except Exception as e: return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/anomalies/delete-all', methods=['POST'])
def delete_all_anomalies():
    try:
        count = 0
        for filename in os.listdir(ANOMALY_LOG_DIR):
            filepath = os.path.join(ANOMALY_LOG_DIR, filename)
            if os.path.isfile(filepath): os.remove(filepath); count += 1
        return jsonify({"status": "success", "message": f"Berhasil menghapus {count} file log anomali."})
    except Exception as e: return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)