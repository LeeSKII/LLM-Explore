import os
import win32com.client
import pywintypes  # 用于捕获com_error

def doc_to_docx_with_error_handling(src_folder, dst_folder, error_folder):
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    if not os.path.exists(error_folder):
        os.makedirs(error_folder)

    for dir_path, dirs, files in os.walk(src_folder):
        for file_name in files:
            file_path = os.path.join(dir_path, file_name)
            base, ext = os.path.splitext(file_name)
            if ext.lower() == '.doc' or ext.lower() == '.docx' and not file_name.startswith('~$'):
                docx_file = os.path.join(dst_folder, base + '.docx')
                if not os.path.exists(docx_file):
                    word = None
                    doc = None
                    try:
                        word = win32com.client.Dispatch("Word.Application")
                        word.Visible = False
                        print(f'Converting: {file_path} to {docx_file}')
                        doc = word.Documents.Open(os.path.abspath(file_path), False, False, False)
                        doc.SaveAs2(os.path.abspath(docx_file), FileFormat=16)
                    except Exception as e:
                        print(f'Failed to convert {file_path}: {e}')
                        # 将异常文件复制到error_folder
                        error_path = os.path.join(error_folder, file_name)
                        shutil.copy2(file_path, error_path)
                    finally:
                        # 安全关闭文档和Word应用，忽略关闭时的异常
                        try:
                            if doc is not None:
                                doc.Close(False)
                        except Exception as e:
                            print(f'Error closing doc: {e}')
                        try:
                            if word is not None:
                                word.Quit()
                        except Exception as e:
                            print(f'Error quitting Word: {e}')
                        

doc_to_docx_with_error_handling(r'C:\Lee\work\contract\全部商务合同',r'C:\Lee\work\contract\全部商务合同docx',r'C:\Lee\work\contract\全部商务合同error')
