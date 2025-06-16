# 根据文档内容生成唯一ID判断是否重复

# 方案一：哈希算法生成内容ID
import hashlib

def generate_content_id(content):
    md5 = hashlib.md5()
    md5.update(content.encode('utf-8'))  # 假设content是字符串；如果是二进制，直接md5.update(content)
    return md5.hexdigest()

# 示例
content = "文档内容示例"
content_id = generate_content_id(content)
print(content_id)  # 输出：类似"e10adc3949ba59abbe56e057f20f883e"

# 方案二：UUID5命名空间方案
import uuid

def generate_content_uuid(content):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, content))

# 示例
content = "文档内容示例"
content_uuid = generate_content_uuid(content)
print(content_uuid)