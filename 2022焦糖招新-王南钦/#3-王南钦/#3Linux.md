# #3Linux



# 1.在虚拟机中安装Ubutun



+ __软件选择：VMware Workstation__

  ~~安装VM这就应该不用说什么了吧~~

  

  发挥个人各种资源手段搞到一个Ubuntu镜像文件

  

  安装Ubuntu

  
  

---



# 2.SSH服务

​       SSH 为 Secure Shell 的缩写，由 IETF 的网络小组（Network Working Group）所制定；SSH 为建立在应用层基础上的安全协议。SSH 是较可靠，专为远程登录会话和其他网络服务提供安全性的协议。利用 SSH 协议可以有效防止远程管理过程中的信息泄露问题。SSH最初是UNIX系统上的一个程序，后来又迅速扩展到其他操作平台。SSH在正确使用时可弥补网络中的漏洞。SSH客户端适用于多种平台。

​      可以通过Windows自带的命令行连接，也可以使用vscode，还可以使用可视化第三方软件如Xshell。

windows命令行：

![](https://s3.bmp.ovh/imgs/2022/09/06/556ebbd732fc0164.png)

Xshell:

![](https://s3.bmp.ovh/imgs/2022/09/06/a5186b93ccfb767f.png)

# 3.通过 `VScode` 的 `Remote` 插件连接至虚拟机

+ __VScode 上安装remote插件__

  

  ![](https://s3.bmp.ovh/imgs/2022/09/02/50f858a93308a522.png)

  此处选择安装Remote Development~~SSH WSL等均一键安装~~

+  **设置要连接的主机IP地址和用户名**



​        ![](https://s3.bmp.ovh/imgs/2022/09/02/0d9063ea4b31c27d.png)

+ **报错~~折磨时刻~~**

  1. **连接的时候报错 A ssh installation not found**

  ​       在服务端和客户端安装ssh服务~~好像win10自带ssh服务~~

  

  2. **提示permission denied**

  ​       打开被连主机的配置文件：**sudo vi /etc/ssh/sshd_config**
  ​       找到PermitRootLogin ，修改为yes
  ​       重启ssh服务

  ​       再次初步认识Ubuntu的终端命令使用

  ​       ~~什么文件权限不够、什么找不到文件在哪里真的把人搞麻了~~

  

  3. __写入管道不存在__

  ​       是配置的时候写错了重新改了一下就好了



# 4.连接成功

​        当VScode出现下图时，就已成功连接

​       ![](https://s3.bmp.ovh/imgs/2022/09/02/28345d70bd7664eb.png)

​       ~~就可以随心所欲地玩耍~~



# 5.免密登录

+ __在客户端生成密匙对__

​       命令：ssh-keygen 可直接生成密匙对，可以连按空格键使密码为空

​       注意密匙对的格式为rsa,在C:\Users\username.ssh 路径下一定有密匙对

+ __将客户端的公钥上传至服务端__

​       这里又要涉及到权限的问题，一是用户的权限，一是文件的权限

​       sudo su 进入root模式

​       

+  __修改文件的权限__

​        改.ssh目录的权限为700，文件authorized_keys和私钥的权限为600

​        chmod 700 目标机器路径/.ssh

​        chmod 600 目标机器路径/.ssh/authorized_keys



+ __测试是否能够免密登录__



# 完成撒花 







































