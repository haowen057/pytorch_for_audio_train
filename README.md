项目名称：基于 Flask + uWSGI + Nginx 的语音关键词识别系统部署
项目简介
本项目实现了一个基于深度学习模型的语音关键词识别（Keyword Spotting）系统，并将其部署为高性能的 Web 服务。系统通过 Flask 框架提供 API 接口，利用 TensorFlow 训练好的模型进行音频文件的关键词识别。为了提升服务的稳定性和并发处理能力，采用 uWSGI 作为 WSGI 服务器，并使用 Nginx 作为反向代理和负载均衡器，实现生产环境的高效部署。

技术栈
深度学习框架：TensorFlow，用于训练和推理语音关键词识别模型（.h5格式模型文件）

Web 框架：Flask，负责处理 HTTP 请求，暴露 /predict 接口，支持上传音频文件

WSGI 服务器：uWSGI，负责高性能地托管 Flask 应用

反向代理服务器：Nginx，负责请求转发、负载均衡和静态资源管理

容器化工具：Docker + Docker Compose，简化环境搭建和多容器协同管理

音频数据：WAV 格式的语音样本，用于关键词检测

功能描述
接收客户端上传的音频文件（wav格式）

调用深度学习模型对音频进行预处理和特征提取

通过模型推理识别关键词

返回识别结果（关键词类别或文本）

Usage Restrictions
-  **Allowed**: Learning, personal projects, non-commercial forks.  
-  **Prohibited**: Commercial use, redistribution without permission.  
