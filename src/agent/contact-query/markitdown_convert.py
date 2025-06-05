from markitdown import MarkItDown

md = MarkItDown(enable_plugins=False) # Set to True to enable plugins
result = md.convert(r"E:\Temp\docx_files\05 变频器及配套高压柜采购合同.docx")
print(result.text_content)