FROM python:3.9-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Set up a new user named "user" with user ID 1000
# (Hugging Face Spaces runs containers as a non-root user)
RUN useradd -m -u 1000 user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy the current directory contents into the container setting the owner to the user
COPY --chown=user . $HOME/app

# Hugging Face Spaces exposes port 7860
EXPOSE 7860

# Run the app using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
