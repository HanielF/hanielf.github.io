---
title: CentOS搭建LNMP环境
comments: true
mathjax: false
date: 2017-08-12 16:12:58
tags: [LNMP,Linux]
categories: Linux
urlname: centos-lnmp
---

<meta name="referrer" content="no-referrer" />

# 前言

woc!因为Algolia文章长度限制...不得不把原本的<CentOS-Fedora搭建LNMP环境>改成两篇文章...佛了
<!--more-->

原本在服务器上一直用的debian系,后来转到本机Fedora搭建环境就蒙了,之前也因为这个花了不少时间,为了把踩过得坑记下来,还是写篇文章。

其实所谓LNMP其实就是指Linux+Nginx+Mysql+PHP,因为Nginx发音问题,有时候也说是LEMP。

**下面就是*Centos/Fedora* 搭建LNMP环境的教程.**  

# 操作环境

OS: Fedora 25  
Nginx Version: 1.6.2  
Mysql Version: MariaDB 10.2  
PHP Version: PHP7.17

# 安装Nginx,PHP7.1.7和PHP-FPM

## 切换到root用户
    
    sudo -i 
    
    ## OR ##
    
    su -  
  
  
## 添加repositories

    ## Centos 7 Remi repo ##
    
    rpm -Uvh https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm
    
    rpm -Uvh http://rpms.famillecollet.com/enterprise/remi-release-7.rpm  
  
    ## Centos 7 Nginx repo ##
    
    vim /etc/yum.repos.d/nginx.repo
    
    ## 添加 ##
    
    [nginx]
    
    name=nginx repo
    
    baseurl=http://nginx.org/packages/centos/$releasever/$basearch/
    
    gpgcheck=0
    
    enabled=1  
  
  
## 安装Nginx,PHP 7.1.7 and PHP-FPM

     yum --enablerepo=remi,remi-php71 install nginx php-fpm php-common  
  
## 安装模块
    
    yum --enablerepo=remi,remi-php71 install php-opcache php-pecl-apcu php-cli php-pear php-pdo php-mysqlnd php-pgsql php-pecl-mongodb php-pecl-redis php-pecl-memcache php-pecl-memcached php-gd php-mbstring php-mcrypt php-xml  
  
## 关闭httpd(Apache)并打开Nginx,PHP-FPM

 **关闭Apache**  

    /etc/init.d/httpd stop
    
    ## OR ## 
    
    service httpd stop  
  
  
**打开Nginx和PHP-FPM**  

    systemctl start nginx.service
    
    systemctl start php-fpm.service  
  
  
## 设置开机自启Nginx PHP-FPM

 **关闭httpd(Apache)的开机自启,并设置nginx和php-fpm开机自启**  

    systemctl disable httpd.service
    
    systemctl enable nginx.service
    
    systemctl enable php-fpm.service  
  
  
## 配置Nginx 和PHP-FPM及其他

这一段的配置和fedora是一样的,可以不看了~

**先做好默认配置的备份**  
    
    cp /etc/nginx/nginx.conf /etc/nginx/nginx.conf.bak
    
    cp /etc/nginx/nginx.conf.default /etc/nginx/nginx.conf  
  
  
**修改PHP-FPM配置**  
    
    vim /etc/php-fpm.d/www.conf
    
    ## 注释掉新添加一行 ##
    
    ;listen = /run/php-fpm/www.sock
    
    listen = 127.0.0.1:9000  
  
**为你的站点创建相关文件夹**  
    
    ## 将testsite.local换成你自己的　##
    
    mkdir -p /srv/www/testsite.local/public_html
    
    mkdir /srv/www/testsite.local/logs
    
    chown -R apache:apache /srv/www/testsite.local
    
    mkdir /etc/nginx/sites-available
    
    mkdir /etc/nginx/sites-enabled  
  
  
**在/etc/nginx/nginx.conf文件”include /etc/nginx/conf.d/*.conf”这行后面添加一行**  
    
    include /etc/nginx/sites-enabled/*;  
  
  
**创建testsite.local 文件并配置**  

    ## 这是最基本的配置 ##
    
    server {
    
        server_name testsite.local;
    
        access_log /srv/www/testsite.local/logs/access.log;
    
        error_log /srv/www/testsite.local/logs/error.log;
    
        root /srv/www/testsite.local/public_html;
    
        location / {
    
            index index.html index.htm index.php;
    
        }
    
        location ~ \.php$ {
    
            include /etc/nginx/fastcgi_params;
    
            fastcgi_pass  127.0.0.1:9000;
    
            fastcgi_index index.php;
    
            fastcgi_param SCRIPT_FILENAME $document_root$fastcgi_script_name;
    
        }
    
    }  
  
  
**创建软链接到sites-enabled目录**  

    cd /etc/nginx/sites-enabled/
    
    ln -s /etc/nginx/sites-available/testsite.local
    
    systemctl restart nginx.service  
  
  
**将testsite.local添加到/etc/hosts中**  
    
    127.0.0.1               localhost.localdomain localhost testsite.local  
  
[更多的Nginx 和PHP-FPM配置点这里～](http://www.if-not-true-then-false.com/2011/nginx-
and-php-fpm-configuration-and-optimizing-tips-and-tricks/)

**测试是否成功**  
创建/srv/www/testsite.local/public_html/index.php并添加以下内容:  

然后访问域名或者服务器ip看是否出现phpinfo的页面。

**出现403 forbidden**  
有可能是SELinux有问题  
    
    chcon -R -t httpd_sys_content_t /srv/www/testsite.local/public_html  
  
  
**远程链接到服务器**  
打开防火墙的80端口  
    
    firewall-cmd --get-active-zones
    
    firewall-cmd --permanent --zone=public --add-service=http
    
    ## OR ##
    
    firewall-cmd --permanent --zone=public --add-port=80/tcp
    
    systemctl restart firewalld.service  

# 安装MariaDB

这里安装的是MariaDB,而不是Mysql,有如下几个原因

  * MariaDB本来就是一个Mysql的开源版本
  * MariaDB和Mysql类似并兼容Mysql
  * Fedora和Centos系列的发行版已经转用MariaDB了

**我这里选择安装MariaDB 10.2.7 [stable],具体安装过程如下~**

  * 如果之前安装了Mysql,记得备份你的数据库和配置文件!
  * 如果是从低版本升级的,记得执行`mysql_upgrade`~

## 切换root用户
    
    su -
    
    ## OR ##
    
    sudo -i  
  
  
## 添加MariaDB repo

现在Fedora 24/25/26 用户都可以直接安装MariaDB 10.1 而不用添加其他的repo来安装~

[MariaDB repository configuration
tool](http://downloads.mariadb.org/mariadb/repositories/),这里面有repo,自己选择repo文件安装,上面也有教程~

    yum install MariaDB MariaDB-server  

## 打开MariaDB并配置自启

    service mysql start ## use restart after update
    
    ## OR ##
    
    /etc/init.d/mysql start ## use restart after update
    
    chkconfig --levels 235 mysql on  
  
## 进行secure installation
    
    /usr/bin/mysql_secure_installation  
  
自己看提示来~第一个没设置密码呢,直接回车,下面的大多数都选Y

如果不想进行secure installation 的话,emmmmm….最好还是做下吧~  
  
到这里其实已经结束了…233333

## 连接数据库
    
    mysql -u root -p  
  
## 创建数据库和user
    
    ## CREATE DATABASE ##
    
    MariaDB [(none)]> CREATE DATABASE webdb;
    
    ## CREATE USER ##
    
    MariaDB [(none)]> CREATE USER 'webdb_user'@'10.0.15.25' IDENTIFIED BY 'password123';
    
    ## GRANT PERMISSIONS ##
    
    MariaDB [(none)]> GRANT ALL ON webdb.* TO 'webdb_user'@'10.0.15.25';
    
    ##  FLUSH PRIVILEGES, Tell the server to reload the grant tables  ##
    
    MariaDB [(none)]> FLUSH PRIVILEGES;  
  
## 确保数据库能远程连接

    vim /etc/sysconfig/iptables-config
    
    ## 添加 ##
    
    -A INPUT -m state --state NEW -m tcp -p tcp --dport 3306 -j ACCEPT
    
    ## 重启 ##
    
    service iptables restart
    
    ## OR ##
    
    /etc/init.d/iptables restart  
  
------------------- 
