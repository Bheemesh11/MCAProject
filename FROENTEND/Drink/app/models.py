from django.db import models

# Create your models here.

from unittest.util import _MAX_LENGTH
from django.db import models

# Create your models here.
from django.db import models
import os

# Create your models here.

class Register(models.Model):
    name=models.CharField(max_length=50)
    email=models.EmailField(max_length=50)
    password=models.CharField(max_length=50)
    age=models.CharField(max_length=50)
    contact=models.CharField(max_length=50)
   




