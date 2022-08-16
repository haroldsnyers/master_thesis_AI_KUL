FROM nvcr.io/nvidia/pytorch:21.11-py3

WORKDIR /app

# By copying over requirements first, we make sure that Docker will cache
# our installed requirements rather than reinstall them on every build
COPY requirements.txt /app/requirements.txt

# RUN /opt/conda/bin/conda install -y nodejs jupyter jupyterlab==2.3.2

RUN pip install -r requirements.txt

#RUN jupyter labextension install jupyterlab_tensorboard
#RUN pip install jupyter_tensorboard
#
## Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.
#ENV TINI_VERSION v0.6.0
#ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
#RUN chmod +x /usr/bin/tini
#ENTRYPOINT ["/usr/bin/tini", "--"]
#
#CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]