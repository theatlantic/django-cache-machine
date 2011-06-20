from django.core.management.base import BaseCommand, CommandError
from caching.invalidation import invalidator

class Command(BaseCommand):
	help = 'Flushes all entries from the django cache'
	
	def handle(self, *args, **options):
		try:
			invalidator.clear()
		except:
			raise CommandError("Failed to flush querycache\n")
		
		self.stdout.write("Successfully flushed querycache\n")