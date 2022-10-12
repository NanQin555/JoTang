import requests
import json
import time

# 创建请求头
headers = {'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36 Edg/106.0.1370.36'}
#可以自己从浏览器获取

fp = open(r'C:\Users\王南钦\Desktop\22-王南钦-10.4-138\#6-王南钦\jingdong.txt', 'w', encoding='utf-8')

# 循环实现翻页写入
for page in range(0, 100):
    # 通过format格式化函数实现翻页功能
    url = 'https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98&productId=100019125569&score=0&sortType=5&page={}&pageSize=10&isShadowSku=0&rid=0&fold=1'.format(
        page)
    print("正在爬起第" + format(page) + "页")
    response = requests.get(url, headers=headers)
    data = response.text
    # 掐头去尾，替换不需要的部分
    # 实际上也可以把url中callback=fetchJSON_comment98去掉
    # lstrip：截去左边的‘’内容；rstrip:截去右边的‘’内容
    jd = json.loads(data.lstrip('fetchJSON_comment98vv12345(').rstrip(');'))
    data_list = jd['comments']
    for data in data_list:
        # 提取数据
        nickname = data['nickname']
        content = data['content']
        content = content.replace('\n', '')
        creationtime = data['creationTime']

        print(nickname)
        print(content)
        print(creationtime)
        print()
        # 写入数据到txt文件中
        fp.write(f'{nickname}\t{content}\t{creationtime}\n')
    # 限制爬取网页速度
    time.sleep(3)

print("完成")

fp.close()