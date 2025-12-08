# Linux常用指令完全指南
这里为你整理一份Linux常用指令分类清单，包括基本操作、文件管理、系统监控等，适合从初学者到高级用户参考使用。

## 📁 一、文件和目录操作

### 基础操作
```bash
# 查看当前目录
pwd                     # 显示当前工作目录

# 切换目录
cd /path/to/dir         # 切换到指定目录
cd ~                    # 切换到用户主目录
cd -                    # 切换到上一个目录
cd ..                   # 返回上级目录

# 列出文件
ls                      # 列出当前目录内容
ls -l                   # 详细列表（权限、大小、时间）
ls -a                   # 显示所有文件（包括隐藏文件）
ls -lh                  # 人类可读的文件大小
ls -t                   # 按时间排序
ls -R                   # 递归列出子目录

# 创建和删除
mkdir dirname           # 创建目录
mkdir -p dir1/dir2      # 创建多级目录
rmdir dirname           # 删除空目录
rm -r dirname           # 递归删除目录及其内容
rm -rf dirname          # 强制递归删除（危险！）
```

### 文件操作
```bash
# 创建文件
touch filename          # 创建空文件或更新文件时间戳

# 复制文件
cp file1 file2          # 复制文件
cp -r dir1 dir2         # 递归复制目录
cp -i file1 file2       # 交互式复制（覆盖前询问）

# 移动/重命名
mv file1 file2          # 移动或重命名文件
mv -i file1 file2       # 交互式移动

# 删除文件
rm filename             # 删除文件
rm -f filename          # 强制删除
rm -i filename          # 交互式删除

# 查看文件
cat filename            # 显示文件全部内容
less filename           # 分页查看文件（支持搜索）
head -n 10 filename     # 显示文件前10行
tail -n 10 filename     # 显示文件后10行
tail -f logfile         # 实时跟踪日志文件更新
```

## 📝 二、文件内容查看和编辑

```bash
# 基本查看
cat file1 file2         # 连接并显示多个文件
more filename           # 分页显示（只能向下）
less filename           # 分页显示（可上下翻页）
nl filename             # 显示行号

# 编辑器
nano filename           # 简单文本编辑器
vim filename            # 功能强大的编辑器
vi filename             # vim的前身

# 统计信息
wc filename             # 统计行数、单词数、字符数
wc -l filename          # 只统计行数

# 文件类型
file filename           # 查看文件类型
```

## 🔍 三、搜索和查找

```bash
# 文件查找
find /path -name "*.txt"        # 按名称查找文件
find . -type f -name "*.log"    # 查找普通文件
find . -size +10M               # 查找大于10M的文件
find . -mtime -7                # 查找7天内修改的文件

# 文件内容搜索
grep "pattern" filename         # 在文件中搜索文本
grep -r "pattern" /path         # 递归搜索目录
grep -i "pattern" filename      # 忽略大小写
grep -n "pattern" filename      # 显示行号
grep -v "pattern" filename      # 反向匹配（不包含的行）
grep -E "regex" filename        # 使用扩展正则表达式

# 命令查找
which command                   # 查找命令的位置
whereis command                 # 查找命令的二进制、源码和手册页
locate filename                 # 快速查找文件（需要updatedb）
```

## 💻 四、系统信息和监控

```bash
# 系统信息
uname -a                        # 显示系统信息
hostname                        # 显示主机名
date                            # 显示日期时间
cal                             # 显示日历
uptime                          # 显示系统运行时间

# 硬件信息
free -h                         # 显示内存使用情况
df -h                           # 显示磁盘空间
du -sh dirname                  # 显示目录大小
lscpu                           # 显示CPU信息
lsblk                           # 显示块设备信息

# 进程管理
ps aux                          # 显示所有进程
ps -ef                          # 显示完整格式的进程
top                             # 实时显示进程状态
htop                            # top的增强版（需安装）
kill PID                        # 终止进程
kill -9 PID                     # 强制终止进程
pkill process_name              # 按名称终止进程

# 系统监控
dmesg                           # 显示内核消息
journalctl                      # 查看系统日志（systemd）
iostat                          # 查看I/O统计
vmstat                          # 查看虚拟内存统计
```

## 🌐 五、网络相关

```bash
# 网络配置和测试
ifconfig                       # 显示网络接口信息（旧）
ip addr show                   # 显示IP地址
ping hostname                  # 测试网络连通性
traceroute hostname            # 跟踪路由路径
netstat -tulpn                 # 显示网络连接和端口
ss -tulpn                      # netstat的替代品

# 下载和传输
wget URL                       # 下载文件
wget -c URL                    # 断点续传
curl URL                       # 传输数据
scp file user@host:path        # 安全复制文件
rsync -av source/ dest/        # 同步文件

# 远程连接
ssh user@hostname              # SSH远程登录
ssh -p port user@hostname      # 指定端口SSH
sftp user@hostname             # 安全文件传输
```

## 📦 六、压缩和归档

```bash
# tar - 打包和压缩
tar -cvf archive.tar dir/      # 打包目录
tar -xvf archive.tar           # 解包
tar -czvf archive.tar.gz dir/  # 打包并gzip压缩
tar -xzvf archive.tar.gz       # 解压gzip压缩包
tar -cjvf archive.tar.bz2 dir/ # 打包并bzip2压缩
tar -xjvf archive.tar.bz2      # 解压bzip2压缩包

# 其他压缩工具
gzip filename                  # 压缩文件（生成.gz）
gunzip filename.gz             # 解压.gz文件
zip -r archive.zip dir/        # 创建zip压缩包
unzip archive.zip              # 解压zip文件
7z x archive.7z                # 解压7z文件
```

## 👥 七、用户和权限管理

```bash
# 用户管理
whoami                         # 显示当前用户
who                            # 显示已登录用户
w                              # 显示已登录用户及其活动
id                             # 显示用户ID和组信息
sudo command                   # 以超级用户权限执行命令
su username                    # 切换用户

# 用户操作
useradd username               # 添加用户
userdel username               # 删除用户
passwd username                # 修改用户密码

# 组管理
groupadd groupname             # 添加用户组
groupdel groupname             # 删除用户组
usermod -aG groupname username # 将用户添加到组

# 权限管理
chmod 755 filename             # 修改文件权限（数字模式）
chmod u+x filename             # 给所有者添加执行权限（符号模式）
chown user:group filename      # 修改文件所有者和组
chgrp group filename           # 修改文件所属组
```

## 🔧 八、包管理（Ubuntu (apt)）

```bash
sudo apt update                # 更新软件包列表
sudo apt upgrade               # 升级所有软件包
sudo apt install package       # 安装软件包
sudo apt remove package        # 删除软件包
sudo apt search keyword        # 搜索软件包
apt list --installed           # 列出已安装的包
```

## 📊 九、文本处理

```bash
# 排序和去重
sort filename                  # 排序文件内容
sort -u filename               # 排序并去重
uniq filename                  # 去除相邻重复行
uniq -c filename               # 统计重复次数

# 剪切和粘贴
cut -d':' -f1 filename         # 按分隔符提取字段
paste file1 file2              # 合并文件列

# 流编辑器
sed 's/old/new/g' file         # 替换文本
sed -n '5,10p' file            # 显示5-10行
sed '/pattern/d' file          # 删除包含模式的行

# 其他工具
awk '{print $1}' file          # 提取第一列
awk -F':' '{print $1}' file    # 指定分隔符提取
tr 'a-z' 'A-Z' < file          # 转换大小写
diff file1 file2               # 比较文件差异
cmp file1 file2                # 比较文件是否相同
```

## 📂 十、磁盘和文件系统

```bash
# 磁盘操作
fdisk -l                       # 列出磁盘分区
mount /dev/sda1 /mnt           # 挂载文件系统
umount /mnt                    # 卸载文件系统
blkid                          # 显示块设备属性
lsblk                          # 列出块设备

# 文件系统检查
fsck /dev/sda1                 # 检查修复文件系统
badblocks /dev/sda1            # 检查磁盘坏道
```

## 🛠️ 十一、其他实用命令

```bash
# 别名和历史
alias ll='ls -l'               # 创建命令别名
history                        # 查看命令历史
!!                             # 执行上一条命令
!n                             # 执行历史中的第n条命令

# 后台和任务
command &                      # 在后台运行命令
jobs                           # 查看后台任务
fg %n                          # 将后台任务调到前台
bg %n                          # 继续后台任务
nohup command &                # 退出终端后继续运行

# 环境变量
echo $PATH                     # 查看PATH变量
export VAR=value               # 设置环境变量
env                            # 显示所有环境变量

# 帮助文档
man command                    # 查看命令手册
command --help                 # 查看命令帮助
info command                   # 查看详细文档
whatis command                 # 显示命令简要说明
```

## 📌 十二、常用快捷键
快捷键	功能
Ctrl + C	终止当前命令
Ctrl + Z	暂停当前命令
Ctrl + D	退出终端/结束输入
Ctrl + L	清屏
Ctrl + A	光标移动到行首
Ctrl + E	光标移动到行尾
Ctrl + U	删除光标前的所有内容
Ctrl + K	删除光标后的所有内容
Ctrl + R	搜索命令历史
Tab	      命令/文件名自动补全

# 🎯 学习建议

## 初级阶段（每天使用）：
ls, cd, pwd, cp, mv, rm, cat, grep, ps, kill

## 中级阶段（每周使用）：
find, tar, ssh, scp, top, df, du, chmod, chown

## 高级阶段（需要时使用）：
awk, sed, crontab, iptables, lsof, strace, tcpdump

记住：多用多练是最好的学习方法。每个命令都可以用 --help 或 man 查看详细用法！
