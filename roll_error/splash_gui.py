import dearpygui.dearpygui as dpg
from spectral.io import envi
import re
import os
import numpy as np
from splash import splash
import subprocess
import sys

# Global variables
vnir_bands = []
swir_bands = []
Mica_bands = []
passed_vnir_bands = []
passed_swir_bands = []
passed_Mica_bands = []
def update_vnir_bands(sender, app_data, user_data):
    global vnir_bands
    hdr_file_path = dpg.get_value("vnir_hdr_path")
    if not hdr_file_path.endswith(".hdr"):
        hdr_file_path = hdr_file_path + ".hdr"
    profile = envi.read_envi_header(hdr_file_path)
    vnir_bands = profile["wavelength"]
    vnir_bands = [f"{c}: {str(i)}"  for c, i in enumerate(vnir_bands)]
    dpg.configure_item("coreg_vnir_bands_combo", items=vnir_bands)


def update_swir_bands(sender, app_data, user_data):
    global swir_bands
    hdr_file_path = dpg.get_value("swir_hdr_path")
    if not hdr_file_path.endswith(".hdr"):
        hdr_file_path = hdr_file_path + ".hdr"
    profile = envi.read_envi_header(hdr_file_path)
    swir_bands = profile["wavelength"]
    swir_bands = [f"{c}: {str(i)}"  for c, i in enumerate(swir_bands)]
    dpg.configure_item("coreg_swir_bands_combo", items=swir_bands)

def update_Mica_bands(sender, app_data, user_data):
    global Mica_bands
    hdr_file_path = dpg.get_value("Mica_hdr_path")
    if not hdr_file_path.endswith(".hdr"):
        hdr_file_path = hdr_file_path + ".hdr"
    profile = envi.read_envi_header(hdr_file_path)
    Mica_bands = profile["band names"]
    Mica_bands = [f"{c}: {str(i)}" for c, i in enumerate(Mica_bands)]
    dpg.configure_item("coreg_Mica_bands_combo", items=Mica_bands)


def separate_text(indices):
    separators = [" , ", ", ", " ,", ","]
    # Create a regex pattern that matches any of the separators
    pattern = '|'.join(map(re.escape, separators))
    # Split the string based on the pattern
    parts = re.split(pattern, indices)
    bands = [int(part) for part in parts if part.isdigit()]
    return bands

def process_vnir_passed_bands(sender, app_data, user_data):
    global passed_vnir_bands
    indices = dpg.get_value("coreg_vnir_band_indices")
    passed_vnir_bands = separate_text(indices)
    print(passed_vnir_bands)
def process_swir_passed_bands(sender, app_data, user_data):
    global passed_swir_bands
    indices = dpg.get_value("coreg_swir_band_indices")
    passed_swir_bands = separate_text(indices)
    print(passed_swir_bands)

def process_Mica_passed_bands(sender, app_data, user_data):
    global passed_Mica_bands
    indices = dpg.get_value("coreg_mica_band_indices")
    passed_Mica_bands = separate_text(indices)
    print(passed_Mica_bands)


class Data:
    def __init__(self):
        self.last_selected = None


d = Data()

def run_splash(sender, app_data):
    vnir_hdr = dpg.get_value("vnir_hdr_path")
    swir_hdr = dpg.get_value("swir_hdr_path")
    mica_hdr = dpg.get_value("Mica_hdr_path")
    out_folder = dpg.get_value("outfolder")

    use_torch = dpg.get_value("use_torch")
    num_threads = dpg.get_value("num_threads")
    manual_warping = dpg.get_value("manual_warping")
    use_homography = dpg.get_value("use_homography")

    pixel_shift = dpg.get_value("pixel_shift")
    kernel_size = dpg.get_value("kernel_size")

    coreg_vnir_band_indices = dpg.get_value("coreg_vnir_band_indices")
    coreg_swir_band_indices = dpg.get_value("coreg_swir_band_indices")
    coreg_mica_band_indices = dpg.get_value("coreg_mica_band_indices")
    # turn them into str
    coreg_vnir_band_indices = map(str, separate_text(coreg_vnir_band_indices))
    coreg_swir_band_indices = map(str, separate_text(coreg_swir_band_indices))
    coreg_mica_band_indices = map(str, separate_text(coreg_mica_band_indices))

    coreg_vnir_band_indices = ' '.join(coreg_vnir_band_indices)
    coreg_swir_band_indices = ' '.join(coreg_swir_band_indices)
    coreg_mica_band_indices = ' '.join(coreg_mica_band_indices)


    ss_vnir_band_start = dpg.get_value("ss_vnir_band_start")
    ss_vnir_band_end = dpg.get_value("ss_vnir_band_end")
    ss_vnir_mica_band_index = dpg.get_value("ss_vnir_mica_band")

    ss_swir_band_start = dpg.get_value("ss_swir_band_start")
    ss_swir_band_end = dpg.get_value("ss_swir_band_end")
    ss_swir_mica_band_index = dpg.get_value("ss_swir_mica_band")

    # command = (
    #     f"python /Volumes/Work/Projects/NURI/NURI/roll_error/splash.py "
    #     f"--vnir_hdr '{vnir_hdr}' "
    #     f"--swir_hdr '{swir_hdr}' "
    #     f"--mica_hdr '{mica_hdr}' "
    #     f"--outfolder '{out_folder}' "
    #     f"--use_torch " if use_torch is True else "" +
    #     f"--num_threads {num_threads} "
    #     f"--manual_warping " if manual_warping is True else "" +
    #     f"--use_homography " if manual_warping is True else "" +
    #     f"--pixel_shift {pixel_shift} "
    #     f"--kernel_size {kernel_size} "
    #     f"--coreg_vnir_band_indices {coreg_vnir_band_indices} "
    #     f"--coreg_swir_band_indices {coreg_swir_band_indices} "
    #     f"--coreg_mica_band_indices {coreg_mica_band_indices} "
    #     f"--ss_vnir_band_start {ss_vnir_band_start} "
    #     f"--ss_vnir_band_end {ss_vnir_band_end} "
    #     f"--ss_vnir_mica_band_index {ss_vnir_mica_band_index} "
    #     f"--ss_swir_band_start {ss_swir_band_start} "
    #     f"--ss_swir_band_end {ss_swir_band_end} "
    #     f"--ss_swir_mica_band_index {ss_swir_mica_band_index}"
    # )

    command = (
            "python /Volumes/Work/Projects/NURI/NURI/roll_error/splash.py " +
            f"--vnir_hdr '{vnir_hdr}' " +
            f"--swir_hdr '{swir_hdr}' " +
            f"--mica_hdr '{mica_hdr}' " +
            f"--outfolder '{out_folder}' " +
            ("--use_torch " if use_torch is True else "") +
            f"--num_threads {num_threads} " +
            (f"--manual_warping " if manual_warping else "") +
            (f"--use_homography " if manual_warping else "") +
            f"--pixel_shift {pixel_shift} " +
            f"--kernel_size {kernel_size} " +
            f"--coreg_vnir_band_indices {coreg_vnir_band_indices} " +
            f"--coreg_swir_band_indices {coreg_swir_band_indices} " +
            f"--coreg_mica_band_indices {coreg_mica_band_indices} " +
            f"--ss_vnir_band_start {ss_vnir_band_start} " +
            f"--ss_vnir_band_end {ss_vnir_band_end} " +
            f"--ss_vnir_mica_band {ss_vnir_mica_band_index} " +
            f"--ss_swir_band_start {ss_swir_band_start} " +
            f"--ss_swir_band_end {ss_swir_band_end} " +
            f"--ss_swir_mica_band {ss_swir_mica_band_index}"
    )

    print(command)
    # command = "python /Volumes/Work/Projects/NURI/NURI/roll_error/temp.py"

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    # Read stdout line by line
    log_file_path = os.path.join(out_folder,"output.log")
    with open(log_file_path, 'w') as log_file:
        for c in iter(process.stdout.readline, b""):
            c = re.compile(r'(?:\x1B[@-_][0-?]*[ -/]*[@-~])').sub('',c)
            dpg.add_text(c, parent="console_window")
            sys.stdout.write(c)
            log_file.write(c)
            # dpg.set_y_scroll("console_window", dpg.get_y_scroll_max("console_window"))
    process.stdout.close()
    process.wait()

    # while True:
    #     line = process.stdout.readline()
    #
    #     if not line:
    #         break  # Break the loop if the line is empty
    #
    #     dpg.add_text(line,
    #                  parent="console_window")
    #     dpg.set_y_scroll("console_window", dpg.get_y_scroll_max("console_window"))
    #
    # # Wait for the process to complete
    #     process.stdout.close()
    #     process.wait()
    #
    #     # Handle any errors
    #     if process.returncode != 0:
    #         error_output = process.stderr.read()
    #         dpg.add_text(f"Error: {error_output}",
    #                      parent="console_window")
    #         dpg.set_y_scroll("console_window", dpg.get_y_scroll_max("console_window"))


def main():
    dpg.create_context()

    with dpg.window(label="Main Window"):
        with dpg.tab_bar():
            with dpg.tab(label="SPLASH"):
                with dpg.collapsing_header(label="Raster HDR Files", default_open = True):
                    dpg.add_text("Please provide the HDR file paths for VNIR and SWIR data after Headwall processing")
                    dpg.add_text("you can add waterfall row to your non-orthorectified using 'WR adder' Tool, prior")
                    dpg.add_text("to using Headwall software.")
                    dpg.add_separator()
                    dpg.add_input_text(label="Output folder", hint="Path to where to the output folder ",
                                       tag="outfolder")
                    dpg.add_input_text(label="VNIR HDR File", hint="Path to VNIR HDR file (after Headwall processing)", tag="vnir_hdr_path", callback=update_vnir_bands, default_value="/Volumes/T7/axhcis/Projects/NURI/data/20210723_tait_labsphere/1133/VNIR/raw_0_rd_wr_or.hdr")
                    dpg.add_input_text(label="SWIR HDR File", hint="Path to SWIR HDR file", tag="swir_hdr_path", callback=update_swir_bands, default_value="/Volumes/T7/axhcis/Projects/NURI/data/20210723_tait_labsphere/1133/SWIR/raw_1504_nuc_rd_wr_or.hdr")
                    dpg.add_input_text(label="Mica HDR File", hint="Path to Mica HDR file (bands stacked)", tag="Mica_hdr_path", callback=update_Mica_bands, default_value="/Volumes/T7/axhcis/Projects/NURI/data/20210723_tait_labsphere/1133/Micasense/NURI_micasense_1133_transparent_mosaic_stacked.hdr")

                with dpg.collapsing_header(label="Co-registration of HSI with Mica", default_open = True):
                    dpg.add_listbox(label="VNIR Bands", items=[], tag="coreg_vnir_bands_combo", num_items=2, width=500)
                    dpg.add_input_text(label="VNIR Band Indices", hint="Enter indices separated by commas",
                                       tag="coreg_vnir_band_indices", width=250, callback=process_vnir_passed_bands, default_value="25, 70, 115")
                    dpg.add_separator()

                    dpg.add_listbox(label="SWIR Bands", items=[], tag="coreg_swir_bands_combo", num_items=2, width=500)
                    dpg.add_input_text(label="SWIR Band Indices", hint="Enter indices separated by commas",
                                       tag="coreg_swir_band_indices", width=250, callback=process_swir_passed_bands, default_value="10, 40, 90")
                    dpg.add_separator()

                    dpg.add_listbox(label="Mica Bands", items=[], tag="coreg_Mica_bands_combo", num_items=2, width=500)
                    dpg.add_input_text(label="Mica Band Indices", hint="Enter indices separated by commas",
                                       tag="coreg_mica_band_indices", width=250, callback=process_Mica_passed_bands, default_value="0, 1, 2")
                    dpg.add_separator()

                    dpg.add_checkbox(label="Use Available Homography", tag= "use_homography", default_value=True)
                    dpg.add_checkbox(label="Manual Warping", tag= "manual_warping", default_value=True)

                with dpg.collapsing_header(label="Shape Shifter and Elastic Registration", default_open=True):
                    with dpg.group(horizontal=True):
                        dpg.add_text("VNIR Band index start")
                        dpg.add_input_int(tag="ss_vnir_band_start", width=100, default_value = 60)
                        dpg.add_text("end")
                        dpg.add_input_int(tag="ss_vnir_band_end", width=100, default_value = 77)
                    with dpg.group(horizontal=True):
                        dpg.add_text("Mica Band index")
                        dpg.add_input_int(tag="ss_vnir_mica_band", width=100, default_value = 1)
                    dpg.add_separator()
                    with dpg.group(horizontal=True):
                        dpg.add_text("SWIR Band index start")
                        dpg.add_input_int(tag="ss_swir_band_start", width=100, default_value = 0)
                        dpg.add_text("end")
                        dpg.add_input_int(tag="ss_swir_band_end", width=100, default_value = 12)
                    with dpg.group(horizontal=True):
                        dpg.add_text("Mica Band index")
                        dpg.add_input_int(tag="ss_swir_mica_band", width=100, default_value = -1)

                    dpg.add_separator()
                    dpg.add_input_int(label="Pixel Shift", tag="pixel_shift",width=100, default_value=3)
                    dpg.add_input_int(label="Kernel Size", tag="kernel_size",width=100, default_value=3)
                    dpg.add_input_int(label="Number of Threads", tag="num_threads",width=100, default_value=os.cpu_count())
                    dpg.add_checkbox(label="Use Torch", tag="use_torch", default_value=True)

                dpg.add_button(label="Run", callback=run_splash)

            with dpg.tab(label="Console"):
                with dpg.group(horizontal=True):
                    pass

                with dpg.child_window(horizontal_scrollbar=True, width=500, height=600, tag="console_window"):
                    pass



    dpg.create_viewport(title='HDR File Input', width=800, height=800)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    main()