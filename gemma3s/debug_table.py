def monitor_and_cleanup_processes():
    while True:
        now = datetime.datetime.now()
        to_remove = []
        for session_id, process_info in user_processes.items():
            if now - process_info["last_active"] > PROCESS_TIMEOUT:
                process_info["queue"].put({"command": "exit"})
                process_info["process"].terminate()
                process_info["process"].join()
                to_remove.append(session_id)
        for session_id in to_remove:
            del user_processes[session_id]
            print(f"Automatically cleaned up process for session {session_id}.")
        time.sleep(10)

def seg_track_app():
    # Only supports gradio==3.38.0
    import gradio as gr
    
    def extract_session_id_from_request(request: gr.Request):
        session_id = hashlib.sha256(f'{request.client.host}:{request.client.port}'.encode('utf-8')).hexdigest()
        # cookies = request.kwargs["headers"].get('cookie', '')
        # session_id = None
        # if '_gid=' in cookies:
        #     session_id = cookies.split('_gid=')[1].split(';')[0]
        # else:
        #     session_id = str(uuid.uuid4())
        print(f"session_id {session_id}")
        return session_id

    def make_editor_value(image):
        """Return a stable drawing-canvas image value."""
        if image is None:
            image = np.zeros((512, 512, 4), dtype=np.uint8)
        return image

    def handle_extract_video_info(session_id, input_video, skip_flag, current_slider_state):
        # If the pipeline already preprocessed the video, skip this handler
        # to avoid resetting the slider/frame back to 0
        if skip_flag:

import json
import os
import sys

# Mocking some parts if needed, but trying direct import first
sys.path.append("/mnt/bb586fde-943d-4653-af27-224147bfba7e/Medgemma/MedSAM2/gemma3s")

try:
    from app_medgemma_new_gradio import load_patient_summary_data, sort_patient_data, create_patient_table_html
except ImportError as e:
    print(f"ImportError: {e}")
    # Fallback: redefine the functions here if import fails due to dependencies
    # Copy-pasting the relevant functions for testing
    
    COMMON_DATA_PATH = "/mnt/bb586fde-943d-4653-af27-224147bfba7e/Medgemma/MedSAM2/common_data"
    from os.path import join, exists
    
    def load_patient_summary_data():
        """Load patient summary data from med_gemma_sample_data directory."""
        sample_data_path = "/mnt/bb586fde-943d-4653-af27-224147bfba7e/Medgemma/MedSAM2/common_data"
        common_format_file = join(sample_data_path, "comman_format.json")
        
        if not exists(common_format_file):
            print(f"[ERROR] Common format file not found: {common_format_file}")
            return []
        
        patient_data = []
        
        try:
            with open(common_format_file, 'r') as f:
                data_list = json.load(f)
            
            for patient_entry in data_list:
                patient_id = patient_entry.get('pid', '')
                
                modalities = []
                patient_dir = join(sample_data_path, patient_id)
                if exists(patient_dir):
                    results_file = join(patient_dir, 'patient_results.json')
                    if exists(results_file):
                        try:
                            with open(results_file, 'r') as rf:
                                results_data = json.load(rf)
                                if 'sess_0' in results_data:
                                    modalities = results_data['sess_0'].get('mod', [])
                        except Exception as e:
                            print(f"[WARNING] Could not read modalities from {results_file}: {e}")
                
                patient_info = {
                    'patient_id': patient_id,
                    'tumor': patient_entry.get('tumor', False),
                    'conf_score': patient_entry.get('conf_score', 0.0),
                    'reviewed_by_radio': patient_entry.get('reviewed_by_radio', False),
                    'remark': patient_entry.get('gemma_hard_coded_remark', 'N/A'),
                    'modalities': ', '.join(modalities) if modalities else 'N/A',
                    'mid_idx': 'N/A'
                }
                patient_data.append(patient_info)
                
        except Exception as e:
            print(f"Error loading patient data from {common_format_file}: {e}")
        
        return patient_data

    def sort_patient_data(patient_data):
        def sort_key(patient):
            reviewed = patient.get('reviewed_by_radio', False)
            has_tumor = patient.get('tumor', False)
            return (reviewed, not has_tumor if reviewed else False)
        return sorted(patient_data, key=sort_key)

    def create_patient_table_html(patient_data):
        if not patient_data:
            return "<p>No patient data available.</p>"
        
        html = """
        <style>
            .patient-table {
                width: 100%;
                border-collapse: collapse;
                font-family: Arial, sans-serif;
                margin: 20px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                table-layout: fixed;
            }
            .patient-table thead {
                background-color: #000000;
                color: white;
            }
            .patient-table th {
                padding: 8px;
                text-align: center;
                font-weight: bold;
                border: 1px solid #ddd;
                font-size: 0.9em;
            }
            .patient-table td {
                padding: 6px 8px;
                border: 1px solid #ddd;
                color: #000000;
                font-size: 0.85em;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
                text-align: center;
            }
            /* Tumor Present + Not Reviewed -> Red (Darker) */
            .patient-table tbody tr.row-tumor-unreviewed {
                background-color: #e57373; 
            }
            /* Tumor Present + Reviewed -> Yellow (Darker) */
            .patient-table tbody tr.row-tumor-reviewed {
                background-color: #fff176;
            }
            /* Normal / No Tumor -> Green (Darker) */
            .patient-table tbody tr.row-normal {
                background-color: #81c784;
            }
    
            .patient-table tbody tr.row-tumor-unreviewed:hover {
                background-color: #ef5350;
            }
            .patient-table tbody tr.row-tumor-reviewed:hover {
                background-color: #ffee58;
            }
            .patient-table tbody tr.row-normal:hover {
                background-color: #66bb6a;
            }
    
            .patient-table th:nth-child(1),
            .patient-table td:nth-child(1) {
                width: 4%;
            }
            .patient-table th:nth-child(2),
            .patient-table td:nth-child(2) {
                width: 13%;
            }
            .patient-table th:nth-child(3),
            .patient-table td:nth-child(3) {
                width: 13%;
            }
            .patient-table th:nth-child(4),
            .patient-table td:nth-child(4) {
                width: 9%;
            }
            .patient-table th:nth-child(5),
            .patient-table td:nth-child(5) {
                width: 9%;
            }
            .patient-table th:nth-child(6),
            .patient-table td:nth-child(6) {
                width: 40%;
            }
            .patient-table th:nth-child(7),
            .patient-table td:nth-child(7) {
                width: 15%;
            }
            .tumor-yes {
                color: #000000;
                font-weight: bold;
            }
            .tumor-no {
                color: #000000;
                font-weight: bold;
            }
            .reviewed-yes {
                color: #000000;
            }
            .reviewed-no {
                color: #000000;
            }
            .conf-score {
                font-weight: bold;
                color: #000000;
            }
            .row-number {
                color: #000000;
                font-size: 0.85em;
            }
        </style>
        <table class="patient-table">
            <thead>
                <tr>
                    <th>#</th>
                    <th>Patient ID</th>
                    <th>Tumor Status</th>
                    <th>Conf Score</th>
                    <th>Reviewed</th>
                    <th>Remark</th>
                    <th>Modalities</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for idx, patient in enumerate(patient_data):
            has_tumor = patient.get('tumor', False)
            is_reviewed = patient.get('reviewed_by_radio', False)
    
            tumor_class = "tumor-yes" if has_tumor else "tumor-no"
            tumor_text = "Tumor Present" if has_tumor else "Normal"
            
            reviewed_class = "reviewed-yes" if is_reviewed else "reviewed-no"
            reviewed_text = "Yes" if is_reviewed else "No"
            
            # Determine row class based on requirements
            if has_tumor:
                if is_reviewed:
                    row_class = "row-tumor-reviewed" # Red -> Yellow
                else:
                    row_class = "row-tumor-unreviewed" # Red
            else:
                row_class = "row-normal" # Green/Default matching "lighter" theme
            
            conf_score = patient.get('conf_score', 0.0)
            
            html += f"""
                <tr class="{row_class}">
                    <td class="row-number">{idx + 1}</td>
                    <td><strong>{patient['patient_id']}</strong></td>
                    <td class="{tumor_class}">{tumor_text}</td>
                    <td class="conf-score">{conf_score:.2f}</td>
                    <td class="{reviewed_class}">{reviewed_text}</td>
                    <td title="{patient.get('remark', '')}">{patient.get('remark', '')}</td>
                    <td title="{patient.get('modalities', '')}">{patient.get('modalities', '')}</td>
                </tr>
            """
        
        html += """
            </tbody>
        </table>
        """
        return html

print("Loading patient data...")
patient_data = load_patient_summary_data()
if not patient_data:
    print("No patient data found!")
else:
    print(f"Found {len(patient_data)} patients.")
    print("First patient:", patient_data[0])

print("Sorting patient data...")
sorted_data = sort_patient_data(patient_data)

print("Generating HTML...")
html = create_patient_table_html(sorted_data)
print("HTML generated successfully. Length:", len(html))
print("First 500 chars of HTML:")
print(html[:500])
