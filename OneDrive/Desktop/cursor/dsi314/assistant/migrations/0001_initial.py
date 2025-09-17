from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name='UploadedDocument',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('original_name', models.CharField(max_length=255)),
                ('file', models.FileField(upload_to='uploads/%Y/%m/%d/')),
                ('text_excerpt', models.TextField(blank=True)),
            ],
        ),
        migrations.CreateModel(
            name='ProcessingRun',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('user_prompt', models.TextField(blank=True)),
                ('ranking_json', models.JSONField(default=list)),
                ('summary_json', models.JSONField(default=list)),
                ('quiz_json', models.JSONField(default=list)),
            ],
        ),
        migrations.AddField(
            model_name='processingrun',
            name='documents',
            field=models.ManyToManyField(related_name='runs', to='assistant.uploadeddocument'),
        ),
    ]

