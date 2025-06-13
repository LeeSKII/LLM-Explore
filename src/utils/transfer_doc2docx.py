import os
import win32com.client
import pythoncom

def doc_to_docx_with_error_handling(src_folder, dst_folder, error_folder):
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    if not os.path.exists(error_folder):
        os.makedirs(error_folder)

    for dir_path, dirs, files in os.walk(src_folder):
        for file_name in files:
            file_path = os.path.join(dir_path, file_name)
            base, ext = os.path.splitext(file_name)
            # 只处理.doc和.docx，且不处理临时文件
            if (ext.lower() == '.doc' or ext.lower() == '.docx') and not file_name.startswith('~$'):
                docx_file = os.path.join(dst_folder, base + '.docx')
                # 如果目标文件已存在则跳过
                if os.path.exists(docx_file):
                    print(f'Skip (already exists): {docx_file}')
                    continue
                word = None
                doc = None
                try:
                    # 初始化COM环境（如果在多线程环境下）
                    pythoncom.CoInitialize()
                    word = win32com.client.Dispatch("Word.Application")
                    word.Visible = False
                    print(f'Converting: {file_path} to {docx_file}')
                    doc = word.Documents.Open(os.path.abspath(file_path), False, False, False)
                    doc.SaveAs2(os.path.abspath(docx_file), FileFormat=16)
                except Exception as e:
                    print(f'Failed to convert {file_path}: {e}')
                    # 将异常文件复制到error_folder
                    try:
                        error_path = os.path.join(error_folder, file_name)
                        shutil.copy2(file_path, error_path)
                    except Exception as copy_err:
                        print(f'Failed to copy error file {file_path}: {copy_err}')
                finally:
                    # 安全关闭文档和Word应用，忽略关闭时的异常
                    try:
                        if doc is not None:
                            doc.Close(False)
                    except Exception as close_err:
                        print(f'Error closing doc: {close_err}')
                    try:
                        if word is not None:
                            word.Quit()
                    except Exception as quit_err:
                        print(f'Error quitting Word: {quit_err}')
                    try:
                        pythoncom.CoUninitialize()
                    except Exception:
                        pass
                        

doc_to_docx_with_error_handling(r'C:\Lee\work\contract\全部商务合同',r'C:\Lee\work\contract\全部商务合同docx',r'C:\Lee\work\contract\全部商务合同error')
