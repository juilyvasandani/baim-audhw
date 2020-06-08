# -*- coding: utf-8 -*-
import scrapy
# import Request module to scrape multiple pages from the craigslist domain
from scrapy import Request

class VacationSpider(scrapy.Spider):
    name = 'vacation'
    allowed_domains = ['craigslist.org']
    start_urls = ['https://chicago.craigslist.org/d/vacation-rentals/search/vac/']

    def parse(self, response):
        deals = response.xpath('//p[@class = "result-info"]')

        for deal in deals :
            title = deal.xpath('a/text()').extract_first()
            # scrape date feature (underneath the p tag with 'result-info' class)
            date = deal.xpath('time[@class="result-date"]/text()').get()
            # scrape price, # of bedrooms and city feature - belonging under two classes (result-info & result-meta)
            rental_price = deal.xpath('span[@class="result-meta"]/span[@class="result-price"]/text()').extract_first("")
            # index bedrooms to only include integer value (not including area)
            bedrooms = deal.xpath('span[@class="result-meta"]/span[@class="housing"]/text()').extract_first("")[20:22]
            # index city to remove parentheses
            neighborhood = deal.xpath('span[@class="result-meta"]/span[@class="result-hood"]/text()').extract_first("")[2:-1]

            lower_rel_url = deal.xpath('a/@href').extract_first()
            lower_url = response.urljoin(lower_rel_url)

            # extract values into a dictionary format
            yield Request(lower_url, callback=self.parse_lower, meta={'Title' : title, 'Date' : date, 'Rental Price' : rental_price, 'Number of Bedrooms' : bedrooms, 'Neighborhood' : neighborhood})

        next_rel_url = response.xpath('//a[@class="button next"]/@href').get()
        next_url = response.urljoin(next_rel_url)

        yield Request(next_url, callback=self.parse)

    def parse_lower(self, response) :
        text = "".join(line for line in response.xpath \
            ('//*[@id="postingbody"]/text()').getall())

        response.meta['Text'] = text
        yield response.meta
