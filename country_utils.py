import pycountry_convert as pc
import pycountry


class country_utils():
    def __init__(self):
        self.d = {}

    def get_dic(self):
        return self.d

    def get_country_details(self, country):
        """Returns country code(alpha_3) and continent"""
        try:
            country_obj = pycountry.countries.get(name=country)
            if country_obj is None:
                c = pycountry.countries.search_fuzzy(country)
                country_obj = c[0]
            continent_code = pc.country_alpha2_to_continent_code(
                country_obj.alpha_2)
            continent = pc.convert_continent_code_to_continent_name(
                continent_code)
            return country_obj.alpha_3, continent
        except:
            if 'Congo' in country:
                country = 'Congo'
            elif country == 'Diamond Princess' or country == 'Laos' or country == 'MS Zaandam'\
                    or country == 'Holy See' or country == 'Timor-Leste':
                return country, country
            elif country == 'Korea, South' or country == 'South Korea':
                country = 'Korea, Republic of'
            elif country == 'Taiwan*':
                country = 'Taiwan'
            elif country == 'Burma':
                country = 'Myanmar'
            elif country == 'West Bank and Gaza':
                country = 'Gaza'
            else:
                return country, country
            country_obj = pycountry.countries.search_fuzzy(country)
            continent_code = pc.country_alpha2_to_continent_code(
                country_obj[0].alpha_2)
            continent = pc.convert_continent_code_to_continent_name(
                continent_code)
            return country_obj[0].alpha_3, continent

    def get_iso3(self, country):
        return self.d[country]['code']

    def get_continent(self, country):
        return self.d[country]['continent']

    def add_values(self, country):
        self.d[country] = {}
        self.d[country]['code'], self.d[country]['continent'] = self.get_country_details(
            country)

    def fetch_iso3(self, country):
        if country in self.d.keys():
            return self.get_iso3(country)
        else:
            self.add_values(country)
            return self.get_iso3(country)

    def fetch_continent(self, country):
        if country in self.d.keys():
            return self.get_continent(country)
        else:
            self.add_values(country)
            return self.get_continent(country)
